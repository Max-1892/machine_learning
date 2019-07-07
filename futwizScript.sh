#!/bin/bash

# Non-rare first
url="http://www.futwiz.com/en/fifa16/players?minrating=1&maxrating=99&release=Legends&page="
pageNum=0
pages=3
while [ ${pageNum} -lt ${pages} ]; do
  page="`wget -qO- ${url}${pageNum}`"
  fullNames="`echo "${page}" | grep '<strong>' | grep -Po '>[\D]+<' | sed 's/>//' | sed 's/<//' | grep -v 'Pack Opener' | sed 's/ /_/g'`"
  overalls="`echo "${page}" | grep "td-sort-results" | grep "label stat" | grep -Po ">[\d]+<" | sed 's/>//' | sed 's/<//'`"
  positions="`echo "${page}" | grep '<td>' | grep -Po ">[\D]+<" | sed 's/>//' | sed 's/<//'`"
  ((pageNum++))

  # Make name array
  index=0
  for name in $fullNames
  do
    namesArray[${index}]="`echo $name | sed 's/_/ /g'`"
    ((index++))
  done
  
  # Make overalls array
  index=0
  for score in $overalls
  do
    overallsArray[${index}]="`echo $score`"
    ((index++))
  done
  
  # Make positions array
  index=0
  for p in $positions
  do
    posArray[${index}]="`echo $p`"
    ((index++))
  done
  
  if [ ${#namesArray[@]} -eq ${#overallsArray[@]} ]; then
    if [ ${#overallsArray[@]} -eq ${#posArray[@]} ]; then
      index=0
      while [ ${index} -lt ${#namesArray[@]} ]; do
        echo "${namesArray[${index}]},${overallsArray[${index}]},${posArray[${index}]}"
        ((index++))
      done
    else
      echo "Wrong sized arrays, part 2"
    fi
  else
    echo "Wrong sized arrays, part 1"  
  fi
done
