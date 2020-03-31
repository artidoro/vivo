#!/bin/bash
for filename in $1"/*.pt;" do
    /bin/bash "./eval.sh $2 $3 $4"
    echo "Results for checkpoint:\n"$path$filename
done