#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 DEST_DIR" >&2
  exit 2
fi

dest=$1
base=https://physionet.org/files/hmc-sleep-staging/1.1
data_base=https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1
mkdir -p "$dest/recordings"
curl --fail --location --silent --show-error "$base/RECORDS" -o "$dest/RECORDS"
curl --fail --location --silent --show-error "$base/SHA256SUMS.txt" -o "$dest/SHA256SUMS.txt"
curl --fail --location --silent --show-error "$base/LICENSE.txt" -o "$dest/LICENSE.txt"

if [[ $(wc -l < "$dest/RECORDS") -ne 151 ]]; then
  echo "Expected 151 EDF records" >&2
  exit 1
fi

files=()
while IFS= read -r record; do
  if [[ ! $record =~ ^recordings/SN[0-9]{3}\.edf$ ]]; then
    echo "Unexpected record name: $record" >&2
    exit 1
  fi
  files+=("$record" "${record%.edf}_sleepscoring.txt")
done < "$dest/RECORDS"

download_one() {
  local relative=$1 expected
  expected=$(awk -v name="$relative" '$2 == name { print $1 }' "$dest/SHA256SUMS.txt")
  if [[ -z $expected ]]; then
    echo "No official checksum for $relative" >&2
    return 1
  fi
  if [[ -f $dest/$relative ]] &&
    printf '%s  %s\n' "$expected" "$dest/$relative" | sha256sum --check --status; then
    return 0
  fi
  curl --fail --location --silent --show-error --retry 5 --max-time 3600 \
    "$data_base/$relative" -o "$dest/$relative.part"
  if ! printf '%s  %s\n' "$expected" "$dest/$relative.part" | sha256sum --check --status; then
    echo "Checksum mismatch after download: $relative" >&2
    return 1
  fi
  mv -f "$dest/$relative.part" "$dest/$relative"
}
export -f download_one
export dest data_base
printf '%s\n' "${files[@]}" | xargs -P 8 -I '{}' bash -c 'download_one "$1"' _ '{}'

for relative in "${files[@]}"; do
  expected=$(awk -v name="$relative" '$2 == name { print $1 }' "$dest/SHA256SUMS.txt")
  printf '%s  %s\n' "$expected" "$dest/$relative"
done | sha256sum --check --status
(cd "$dest" && grep -E ' (RECORDS|LICENSE\.txt)$' SHA256SUMS.txt | sha256sum --check --status)
echo "Verified 151 HMC v1.1 EDF and sleep scoring text pairs against PhysioNet SHA256SUMS.txt"
