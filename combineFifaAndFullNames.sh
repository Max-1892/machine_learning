#!/bin/bash
index=0
nopArray=()
fullLines=()
while read line; do
  nOP="`echo "${line}" | grep -Po "^[\D]+,[\d]+,[\D]+," | sed 's/,$//'`"
  nopArray[${index}]="${nOP}"
  fullLines[${index}]="${line}"
  ((index++))
done < Full_List_Without_Accents.txt

index=0
noMatchCount=0
for nop in "${nopArray[@]}"; do
   fullName="`grep -Pzo ".*${nop};" FNOPF.txt.copy`"
   count="`grep -Pzco ".*${nop};" FNOPF.txt.copy`"
   if [ "${count}" -eq "1" ]; then
     #echo "${nop} is really ${fullName}"
     # Update full line
     editedName="`echo "${fullName}" | sed 's/;/,/g'`"
     line="${fullLines[${index}]}"
     editedLine="`echo "${line}" | sed -E "s/[A-Z|a-z| |-]+,[0-9]+,[A-Z|a-z]+,//g"`"
     echo "${editedName}${editedLine}"
   elif [ "${count}" -eq "0" ]; then
     echo "${fullLines[${index}]}"
     ((noMatchCount++))
   fi
   ((index++))
done

echo "Didn't find matches for ${noMatchCount} players"
