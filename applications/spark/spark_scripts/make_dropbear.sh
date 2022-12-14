#!/bin/bash
mkdir dropbear
cd dropbear
dropbearkey -t rsa -s 4096 -f dropbear_rsa_host_key
dropbearkey -t dss -s 1024 -f dropbear_dss_host_key
dropbearkey -t ecdsa -s 521 -f dropbear_ecdsa_host_key
