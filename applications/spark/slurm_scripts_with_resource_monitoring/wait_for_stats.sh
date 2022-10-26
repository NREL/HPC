#!/bin/bash

while [ $(pgrep -f "jade stats collect") ];
do
    sleep 1
done
