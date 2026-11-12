"""Vendors hidden from a database role, for an MCP key that must never see them.

``restrict`` (``vendorkb access restrict``, run as the owner) makes a login role that can read
what the MCP tools read and nothing else, and records the vendors hidden from it in
``access_hidden_vendors``. The row-level-security policies of ``schema.sql`` then hide those
vendors' documents, chunks, claims, facts and offerings from every query the role runs.

``check_restricted`` is the MCP server's start-up proof (``mcp_app``): the restricted
connection is no superuser, cannot bypass row-level security, hides at least the vendors the
settings name, and sees none of their rows. The server refuses to start otherwise.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import psycopg
from psycopg import sql

from vendorkb.db import connect

# What the MCP tools read (``mcp_backend``): the corpus, the taxonomy and the facts. The sync,
# extraction and mapping tables stay unreadable.
READ_TABLES = ("documents", "chunks", "raw_claims", "capabilities", "dimensions", "classes",
               "object_types", "targets", "variants", "offerings", "editions", "aliases",
               "topics", "topic_aspects", "taxonomy_meta", "embedding_cache")
# rows that name a vendor directly: the start-up check counts the hidden vendors' rows here
VENDOR_TABLES = ("documents", "raw_claims", "capabilities", "offerings")
_ROLE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def parse_vendors(text: str) -> list[str]:
    """``"Cohesity, Foo"`` → ``["Cohesity", "Foo"]`` (blank entries dropped)."""
    return [v.strip() for v in text.split(",") if v.strip()]


async def restrict(conn: psycopg.AsyncConnection, role: str, hidden: Iterable[str],
                   password: str | None) -> None:
    """Create (with ``password``) or update the login role ``role`` and hide ``hidden`` from
    it, replacing the vendors it hid before. A new role needs a password; an existing one keeps
    its own unless one is given. Run as the owner, after ``init_schema``."""
    if not _ROLE.match(role):
        raise ValueError(f"role must be a plain lowercase identifier: {role!r}")
    vendors = sorted({v.strip() for v in hidden if v.strip()})
    if not vendors:
        raise ValueError("name at least one vendor to hide")
    ident = sql.Identifier(role)
    async with conn.transaction():
        cur = await conn.execute("SELECT rolsuper OR rolbypassrls FROM pg_roles"
                                 " WHERE rolname = %s", (role,))
        row = await cur.fetchone()
        if row is None:
            if not password:
                raise ValueError(f"role {role} does not exist: give it a password")
            await conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB"
                                       " NOCREATEROLE PASSWORD {}").format(ident,
                                                                          sql.Literal(password)))
        elif row[0]:
            raise ValueError(f"role {role} is a superuser or bypasses row-level security")
        elif password:
            await conn.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                ident, sql.Literal(password)))
        await conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            sql.Identifier(conn.info.dbname), ident))
        cur = await conn.execute("SELECT current_schema()")
        schema = sql.Identifier((await cur.fetchone())[0])  # type: ignore[index]
        await conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, ident))
        await conn.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(
            sql.SQL(", ").join(sql.Identifier(t) for t in READ_TABLES), ident))
        # the query-embedding cache (search.embed_queries): an embedding, no vendor data
        await conn.execute(sql.SQL("GRANT INSERT ON embedding_cache TO {}").format(ident))
        await conn.execute("DELETE FROM access_hidden_vendors WHERE role_name = %s", (role,))
        async with conn.cursor() as c:
            await c.executemany(
                "INSERT INTO access_hidden_vendors (role_name, vendor) VALUES (%s, %s)",
                [(role, v) for v in vendors])


async def check_restricted(dsn: str, hidden: Iterable[str]) -> str:
    """Raise RuntimeError unless ``dsn`` connects as a role that row-level security applies to,
    that hides every vendor of ``hidden`` and sees no row of them. Returns the role."""
    want = {v.strip().lower() for v in hidden if v.strip()}
    conn = await connect(dsn)
    try:
        cur = await conn.execute("SELECT r.rolname, r.rolsuper, r.rolbypassrls,"
                                 " vendorkb_hidden_vendors() FROM pg_roles r"
                                 " WHERE r.rolname = session_user")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("restricted connection: its role is not found")
        role, superuser, bypass, hides = row
        if superuser or bypass:
            raise RuntimeError(f"restricted connection: role {role} bypasses row-level security")
        if missing := sorted(want - set(hides)):
            raise RuntimeError(f"restricted connection: role {role} does not hide {missing}"
                               " (run vendorkb access restrict)")
        for table in VENDOR_TABLES:
            cur = await conn.execute(sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM {} WHERE lower(vendor) = ANY(%s))"
            ).format(sql.Identifier(table)), (sorted(want),))
            if (await cur.fetchone())[0]:  # type: ignore[index]
                raise RuntimeError(f"restricted connection: hidden vendors' rows in {table}")
        return str(role)
    finally:
        await conn.close()
