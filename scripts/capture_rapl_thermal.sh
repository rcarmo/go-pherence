#!/usr/bin/env bash
# Capture bounded host RAPL counters and thermal-zone temperatures as JSONL.
set -euo pipefail

usage() {
  echo "usage: $0 OUTPUT.jsonl COMMAND [ARG ...]" >&2
  exit 2
}
[[ $# -ge 2 ]] || usage
output=$1
shift
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 1; }
command -v jq >/dev/null
command -v sudo >/dev/null
sudo -n true

read_snapshot() {
  local kind=$1 timestamp zone name energy maximum temp
  timestamp=$(date +%s%N)
  printf '{"kind":"%s","timestamp_ns":%s,"rapl":[' "$kind" "$timestamp"
  local first=1
  for zone in /sys/devices/virtual/powercap/intel-rapl/intel-rapl:*; do
    [[ -r "$zone/name" && -r "$zone/max_energy_range_uj" ]] || continue
    name=$(<"$zone/name")
    maximum=$(<"$zone/max_energy_range_uj")
    energy=$(sudo -n cat "$zone/energy_uj")
    [[ $name =~ ^[A-Za-z0-9._-]+$ && $energy =~ ^[0-9]+$ && $maximum =~ ^[0-9]+$ ]] || { echo "invalid RAPL value" >&2; return 1; }
    (( first )) || printf ','
    first=0
    jq -cn --arg name "$name" --argjson energy "$energy" --argjson maximum "$maximum" '{name:$name,energy_uj:$energy,max_energy_range_uj:$maximum}'
  done
  printf '],"thermal":['
  first=1
  for zone in /sys/class/thermal/thermal_zone*; do
    [[ -r "$zone/type" && -r "$zone/temp" ]] || continue
    name=$(<"$zone/type")
    temp=$(<"$zone/temp")
    [[ $name =~ ^[A-Za-z0-9._-]+$ && $temp =~ ^-?[0-9]+$ ]] || { echo "invalid thermal value" >&2; return 1; }
    (( first )) || printf ','
    first=0
    jq -cn --arg name "$name" --argjson temp "$temp" '{name:$name,temp_millicelsius:$temp}'
  done
  printf ']}\n'
}

start=$(read_snapshot start)
set +e
"$@"
status=$?
set -e
end=$(read_snapshot end)
printf '%s\n%s\n' "$start" "$end" | jq -c . > "$output"
jq -cn --argjson start "$start" --argjson end "$end" --argjson command_status "$status" '
  def indexed(xs): reduce xs[] as $x ({}; .[$x.name]=$x);
  def delta($a;$b): if $b.energy_uj >= $a.energy_uj then $b.energy_uj-$a.energy_uj else $a.max_energy_range_uj-$a.energy_uj+$b.energy_uj end;
  (indexed($start.rapl)) as $before | (indexed($end.rapl)) as $after |
  {kind:"summary",command_status:$command_status,elapsed_ns:($end.timestamp_ns-$start.timestamp_ns),energy_uj:[$before|keys[] as $name|{name:$name,delta_uj:delta($before[$name];$after[$name])}],thermal_start:$start.thermal,thermal_end:$end.thermal}' >> "$output"
exit "$status"
