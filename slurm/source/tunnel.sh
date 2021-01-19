:<<++++

Author: Tim Kaiser

Bash function, tunnel, to set up a ssh tunnel to connect to a jupyter 
notebook running on Eagle or a compute node.

To install tunnel cut off this header and add the remains in your .bashrc 
file on your DESKTOP MACHINE NOT EAGLE or save it to dotunnel and do a 
source dotunnel.

Usage:

tunnel NODE_NAME PORT

where NODE_NAME is the node on which you are running.
PORT is the number, usually 8888 returned by jupyter.
After running this function copy the http://... string 
returned by jupyter to your local browser. 
 
If you get a message of the form:
bind [127.0.0.1]:8888: Address already in use
You can close the port.
Run the following command on your linux/Mac :
lsof -i TCP:8888 | grep ssh | awk '{print $2}' | head -1
This will give the process id using the port, say 12768.
Then run the kill command on the process, for example:
     kill 12768
If you do not want to close the port you can run this command
and specify a new local port, say 8889
     tunnel el1 8888 8895
Then when you copy the string from jupyter replace 8888 with 8895.

++++


tunnel() {
# Bash function to set up a ssh tunnel to connect to a jupyter 
# notebook running on Eagle or a compute node.
# 
# To use this add it in your .bashrc file on your DESKTOP MACHINE
# NOT EAGLE  or save it to dotunnel and do a source dotunnel. 
#
# Takes two arguments, the node running our notebook and port number
# from the http string returned by jupyter.
# 
# Note: the -t option for the ssh command allows an interactive 
# session on the node to which we are connecting.
#
# Check for the correct number of arguments.
    if [ $# -lt 2 ]; then 
        echo "Usage:"
        echo "tunnel NODE_NAME PORT"
        echo "tunnel -help"
        if [[ $1 = -h* ]] ; then
          echo "where NODE_NAME is the node on which you are running."
		  echo "PORT is the number, usually 8888 returned by jupyter."
		  echo "After running this function copy the http://... string "
		  echo "returned by jupyter to your local browser. "
		  echo " "
		  echo "If you get a message of the form:"
		  echo "bind [127.0.0.1]:8888: Address already in use"
		  echo "You can close the port."
		  echo "Run the following command on your linux/Mac :"
		  echo "lsof -i TCP:8888 | grep ssh | awk '{print $2}' | head -1"
		  echo "This will give the process id using the port, say 12768."
		  echo "Then run the kill command on the process, for example:"
		  echo "     kill 12768"
		  echo "If you don't want to close the port you can run this command"
		  echo "and specify a new local port, say 8889"
		  echo "     tunnel el1 8888 8895"
		  echo "Then when you copy the string from jupyter replace 8888 with 8895."
		fi
        return -1
    fi
# Check that the second argument in an integer
    echo "$2" | grep -E ^\-?[0-9]+$ > /dev/null
    if [ $? -ne 0 ]  ; then 
        echo "Usage:"
        echo "tunnel NODE_NAME PORT"
        echo "tunnel -help"
        return -1
	fi
	port=$2
	node=$1
	lport=$port
	if [ $# -eq 3 ]; then
	    lport=$3
	fi   
# If we are running on a login node (starts with e) then use the 
# short form of the tunnel, no hop. 
	if [[ $node = e* ]] ; then 
		echo "Running:"
		echo ssh -t -L $lport:localhost:$port $node.hpc.nrel.gov
			 ssh -t -L $lport:localhost:$port $node.hpc.nrel.gov
	else
# If we are running on a compute node (does not start with e) then use the 
# long form of the tunnel. We pick a random intermediate port for the tunnel.
		#let p=`jot -r 1 8100 9100`
		addto=$(( $RANDOM % 1000 ))
		p=$(( 8100 + $addto))

		echo "Running:"
		echo ssh -t -L $lport:localhost:$p eagle ssh -L $p:localhost:$port $node
			 ssh -t -L $lport:localhost:$p eagle ssh -L $p:localhost:$port $node
	fi
}
