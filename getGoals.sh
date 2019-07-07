#!/bin/bash

# For each page of the league, grab the names and goal tallies for each player
#url="http://www.transfermarkt.co.uk/premier-league/torschuetzenliste/wettbewerb/GB1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/ligue-1/torschuetzenliste/wettbewerb/FR1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/serie-a/torschuetzenliste/wettbewerb/IT1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/1-bundesliga/torschuetzenliste/wettbewerb/L1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/la-liga/torschuetzenliste/wettbewerb/ES1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/saudi-professional-league/torschuetzenliste/wettbewerb/SA1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/allsvenskan/torschuetzenliste/wettbewerb/SE1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/bundesliga/torschuetzenliste/wettbewerb/A1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/jupiler-pro-league/torschuetzenliste/wettbewerb/BE1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/campeonato-brasileiro-serie-a/torschuetzenliste/wettbewerb/BRA1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/2-bundesliga/torschuetzenliste/wettbewerb/L2/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/primera-division-clausura/torschuetzenliste/wettbewerb/CHL1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/eredivisie/torschuetzenliste/wettbewerb/NL1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/league-one/torschuetzenliste/wettbewerb/GB3/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/league-two/torschuetzenliste/wettbewerb/GB4/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/championship/torschuetzenliste/wettbewerb/GB2/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/super-league/torschuetzenliste/wettbewerb/GR1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/a-league/torschuetzenliste/wettbewerb/AUS1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/serie-b/torschuetzenliste/wettbewerb/IT2/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/k-league-classic/torschuetzenliste/wettbewerb/RSK1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/segunda-division/torschuetzenliste/wettbewerb/ES2/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/liga-mx-apertura/torschuetzenliste/wettbewerb/MEXA/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/liga-postobon-i/torschuetzenliste/wettbewerb/COLP/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/ligue-2/torschuetzenliste/wettbewerb/FR2/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/major-league-soccer/torschuetzenliste/wettbewerb/MLS1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/primeira-liga/torschuetzenliste/wettbewerb/PO1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/primera-division/torschuetzenliste/wettbewerb/AR1N/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/raiffeisen-super-league/torschuetzenliste/wettbewerb/C1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/premier-liga/torschuetzenliste/wettbewerb/RU1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/scottish-premiership/torschuetzenliste/wettbewerb/SC1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/absa-premiership/torschuetzenliste/wettbewerb/SFA1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/league-of-ireland/torschuetzenliste/wettbewerb/IR1/saison_id/2014/"
#url="http://www.transfermarkt.co.uk/alka-superligaen/torschuetzenliste/wettbewerb/DK1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/ekstraklasa/torschuetzenliste/wettbewerb/PL1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/tippeligaen/torschuetzenliste/wettbewerb/NO1/saison_id/2014/detailpos//altersklasse/alle/plus//"
#url="http://www.transfermarkt.co.uk/super-lig/torschuetzenliste/wettbewerb/TR1/saison_id/2014/detailpos//altersklasse/alle/plus//"
url="http://www.transfermarkt.co.uk/premier-liga/torschuetzenliste/wettbewerb/UKR1/saison_id/2014/detailpos//altersklasse/alle/plus//"

pageNum=1
pageNumStr="page/"
iterations=100
while [ $pageNum -le $iterations ]
do
    #echo "Page number: ${pageNum}"
    page="`wget -qO- ${url}${pageNumStr}${pageNum}`"
    # Grab names, replaces spaces with underscores to keep names together
    names="`wget -qO- ${url}${pageNumStr}${pageNum} | grep 'leistungsdaten\/spieler'| grep "title" | sed "s/&#039;/\'/" | grep -Po '<a title=\"[\D]+\"' | uniq | sed -e 's/<a title=\"//;s/\" class="" id=\"//' | sed 's/ /_/g'`"

    # when we see the same list of names, we are done
    if [[ ${lastNames} == ${names} ]]; then
      #echo "Last page"
      break
    fi

    # Replace underscores with spaces and add to array
    nameIndex=0
    for name in $names
    do
      #echo $name
      namesArray[nameIndex]="`echo $name | sed 's/_/ /g'`"
      ((nameIndex++))
    done
    #echo "${#namesArray[@]}"

    # Grab goal tallies
    goals="`wget -qO- ${url}${pageNumStr}${pageNum} | grep 'leistungsdaten\/spieler' | grep "title" | grep -Po '[\d]+<\/a><\/td><\/tr>' | sed 's/<\/a><\/td><\/tr>//'`"

    # Create goal array
    goalIndex=0
    for goal in $goals
    do
      goalsArray[goalIndex]=$goal
      ((goalIndex++))
    done
    #echo "${#goalsArray[@]}"

    # Combine names and goals
    if [ "${#namesArray[@]}" == "${#goalsArray[@]}" ]; then
      idx=0
      while [ $idx -lt "${#namesArray[@]}" ]
      do
        echo "${namesArray[idx]},${goalsArray[idx]}"
        ((idx++))
      done
    else
      echo "Different number of names and goals"
    fi

    unset goalsArray
    unset namesArray
    lastNames=${names}
    ((pageNum++))
  
done
