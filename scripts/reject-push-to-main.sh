#!/usr/bin/env bash
# Hook de pre-push: recusa qualquer push cujo destino seja a branch `main`.
#
# Instalado direto em .git/hooks/pre-push por `make setup` (não via pre-commit:
# o pre-commit não repassa o stdin e pula o hook quando não há commit novo, o
# que deixava passar `git push origin develop:main`).
#
# O git envia, via stdin, uma linha por ref sendo empurrada:
#   <local_ref> <local_sha> <remote_ref> <remote_sha>
#
# Para emergência, dá pra pular com `git push --no-verify` (o workflow
# "Guard: no direct push to main" denuncia).
set -euo pipefail

protected="refs/heads/main"

while read -r _local_ref _local_sha remote_ref _remote_sha; do
  if [ "$remote_ref" = "$protected" ]; then
    echo "" >&2
    echo "Push direto na 'main' bloqueado." >&2
    echo "   Fluxo correto: branch -> PR -> develop -> PR -> main." >&2
    echo "   (Emergência: 'git push --no-verify'. O CI vai denunciar.)" >&2
    echo "" >&2
    exit 1
  fi
done

exit 0
