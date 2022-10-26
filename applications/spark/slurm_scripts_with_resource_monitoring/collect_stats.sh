#!/bin/bash

if [ -z $1 ]; then
    echo "Error: CONFIG_DIR must be passed to collect-stats.sh"
    exit 1
fi
config_dir=$(realpath ${1})
out_dir=${config_dir}/stats-$(hostname)
jade stats collect -i1 -o ${out_dir} --force &
while ! [ -f shutdown ];
do
    sleep 5
done

for pid in $(pgrep -f "jade stats collect");
do
    kill -TERM ${pid}
done
jade stats plot -o ${out_dir}
