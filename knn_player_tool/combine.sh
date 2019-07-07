#!/bin/bash

# Read in players and store in array
index=0
fifaNames=()
fullLines=()
while read line; do
  fifaName="`echo "${line}" | grep -Po "^[\D]+," | sed 's/,//'`"
  fifaNames[${index}]="${fifaName}"
  fullLines[${index}]="${line}"
  ((index++))
done < Full_Names_All_Ratings_No_Legends_No_Gk.txt

# Read in goal scorers
index=0
fullNames=()
goalsScored=()
scorerLine=""
while read line; do
  fullName="`echo "${line}" | grep -Po "^[\D]+," | sed 's/,//'`"
  fullNames[${index}]="${fullName}"
  goals="`echo "${line}" | grep -Po ",[\d]+$" | sed 's/,//'`"
  goalsScored[${index}]="${goals}"
  scorerLine+="${fullName};"
  ((index++))
done < "$1"

# Compare FIFA data to goal scorers
lineIndex=0
for fifaName in "${fifaNames[@]}"; do
  occurences="`grep -Po ";${fifaName};" <<< "${scorerLine}" | grep -o "${fifaName}" | wc -l`"
  if [ "${occurences}" -eq "1" ]; then
    # Determine index scorerLine
    index="`grep -Po ".*${fifaName};" <<< "${scorerLine}" | grep -o ';' | wc -l`"
    # Get goals with that index
    # Add goals to end of output line
    echo "${fullLines[ ${lineIndex} ]},${goalsScored[ ${index} - 1 ]}"
  elif [ "${occurences}" -eq "0" ]; then
    echo "${fullLines[${lineIndex}]},0" 
  fi
  ((lineIndex++))
done
