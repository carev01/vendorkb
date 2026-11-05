{{- define "vendorkb.labels" -}}
app.kubernetes.io/part-of: vendorkb
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "vendorkb.image" -}}
{{- $tag := required "image.tag is required (sha-<7-char commit>)" .Values.image.tag -}}
{{- if not (regexMatch "^sha-[0-9a-f]{7}$" $tag) -}}
{{- fail (printf "image.tag %q must match the expected format sha-<7-char lowercase-hex commit>, e.g. sha-abc1234" $tag) -}}
{{- end -}}
{{ .Values.image.repository }}:{{ $tag }}
{{- end }}

{{- define "vendorkb.envFrom" -}}
envFrom:
  - secretRef:
      name: {{ .Values.secretName }}
  - configMapRef:
      name: vendorkb-config
{{- end }}

{{/* POSTGRES_PASSWORD must be defined before VENDORKB_DSN, which expands it. */}}
{{- define "vendorkb.env" -}}
env:
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef: {name: {{ .Values.secretName }}, key: POSTGRES_PASSWORD}
  - name: VENDORKB_DSN
    value: {{ printf "postgresql://%s:$(POSTGRES_PASSWORD)@vendorkb-postgres:5432/%s" .Values.postgres.user .Values.postgres.database | quote }}
{{- end }}

{{- define "vendorkb.pullSecrets" -}}
{{- with .Values.imagePullSecrets }}
imagePullSecrets: {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "vendorkb.podSecurity" -}}
automountServiceAccountToken: false
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  seccompProfile: {type: RuntimeDefault}
{{- end }}

{{- define "vendorkb.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
{{- end }}
