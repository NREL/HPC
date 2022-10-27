#!/bin/bash

while [ $(pgrep -f "collect_stats") ];
do
    sleep 1
done
