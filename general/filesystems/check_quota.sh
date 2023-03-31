#!/bin/bash

check_quota()
{
    my_projects=$(groups)

    echo "Project       ID      Used        Quota     Limit"
    echo "-------------------------------------------------"

    for proj in $my_projects; do

        # Let's check if the group/project exists
        error=$(lfs project -d /projects/$proj 2>&1 >/dev/null)
        if [[ -n "$error" ]]; then continue; fi

        proj_id=$(lfs project -d /projects/$proj | cut -f2 -d " ")
        proj_all=$(lfs quota -hp $proj_id /projects/$proj | grep "T")

        if [ ${#proj} -lt 7 ]; then
            # proj name is short
            proj_used=$(echo $proj_all  | cut -f2 -d " ")
            proj_quota=$(echo $proj_all | cut -f3 -d " ")
            proj_limit=$(echo $proj_all | cut -f4 -d " ")
        else
            # proj name is long
            proj_used=$(echo $proj_all  | cut -f1 -d " ")
            proj_quota=$(echo $proj_all | cut -f2 -d " ")
            proj_limit=$(echo $proj_all | cut -f3 -d " ")
        fi

        printf "%-12s  %-6s  %-10s  %-8s  %-10s\n" "$proj" "$proj_id" "$proj_used" "$proj_quota" "$proj_limit"

    done
}
