#!/bin/bash

#Moves config file with sftp to remote nodes and starts them

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Starts the players on remote servers -- [!] run after setup.sh"
   echo "Make sure that you have already the servers in the ~/.ssh/known_hosts file"
   echo "Syntax: setup.sh [-options]"
   echo "options:"
   echo "c     initial id of first player server in cluster, e.g if id goes from 76 to 78 and 76 is master/client, then it's 77"
   echo "n     num of parties players, use 1 if client-server"
   echo "u     ssh username"
   echo "p     ssh password"
   echo "l     logN value, 14 or 15"
   echo "m     model, crypto or nn"
   echo "z     nn value, 20 or 50"
   echo "s     subnet of cluster, eg. 10.90.40"
   echo "a     initial address of cluster, e.g if player 1 is 10.90.40.[2], then is 2 (assuming player 2 is at 10.90.40.3 and so on)"
   echo "h     help"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
while getopts ":c:n:u:p:l:m:z:s:a:h" flag; do
    case "${flag}" in
        c) startingId=${OPTARG};;
        n) parties=${OPTARG};;
        u) username=${OPTARG};;
        p) password=${OPTARG};;
        l) logN=${OPTARG};;
        m) model=${OPTARG};;
        z) nn=${OPTARG};;
        s) subnet=${OPTARG};;
        a) addr=${OPTARG};;
        h) Help
           exit;;
    esac
done

i=0
while [ $i -lt $parties ]; do
  id=$((startingId+i))
  address=$subnet".""$addr"
  echo "connecting to: ${username}:${password}@iccluster0${id}.iccluster.epfl.ch at address ${address}"
  sshpass -p 1 ssh root@iccluster0"${id}".iccluster.epfl.ch "cd /root/dnn/config; chmod +x inference; ./inference --nn ${nn} --model ${model} --logN ${logN} --addr ${address};" &
  i=$((i+1))
  addr=$((addr+1))
done