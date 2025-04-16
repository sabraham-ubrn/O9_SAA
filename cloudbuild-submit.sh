#!/usr/bin/env bash
#
# BRANCH_NAME and SHORT_SHA are automatically provided by Cloud Build when building via trigger.
# Here we are submitting manually, so we'll pass these in.
#
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
SHORT_SHA="$(git rev-parse --short HEAD)"
remote="${1}"

substitutions="SHORT_SHA=${SHORT_SHA},BRANCH_NAME=${BRANCH_NAME}"
if [[ -n "${remote}" ]]; then
  gcloud builds submit --substitutions="${substitutions}"
else
  cloud-build-local --dryrun=false --substitutions="${substitutions}" .
fi