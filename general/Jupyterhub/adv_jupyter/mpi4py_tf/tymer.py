#!/usr/bin/env python
import sys
import time
import os.path
global doquit
doquit=False

import io 
def sprint(*args, **kwargs): 
    sio = io.StringIO() 
    print(*args, **kwargs, file=sio) 
    return sio.getvalue() 
     
def tymer(sys_argv=""):
	secs=time.time()
	date=time.asctime()
	delta=0
	delta2=0
	comment=""
	file=""
	if doquit==False:
		if isinstance(sys_argv,list):
			sys_argv=["dummy"]+sys_argv
		else:
			sys_argv=["dummy"]+[sys_argv]
	if len(sys_argv) > 2:
		for x in sys_argv[2:]:
			comment=comment+x+" "
		comment=comment.strip()

	if len(sys_argv) > 1:
		file=sys_argv[1]
	else:
		s="%17.6f %s %10.3f" % (secs,date,delta)
		print(s)
		if doquit:
			quit()
	if len(file) == 0:
		s="%17.6f %s %10.3f" % (secs,date,delta)
		print(s+" "+comment)
		return(None)		
	if file == "-h" or file == "-help" or file == "--h" :
		print("")
		print("USAGE:")
		print("")
		print(sys_argv[0]," [file] [comment]")
		print("")
		print("With no input on the command line prints")
		print("time in seconds and date to the screen.")
		print("")
		print("With a file name on the command line it reads")
		print("the file, if it exists, and prints a delta time")
		print("from the last time this program updated the file")
		print("and appends the time information to the file.")
		print("")
		print("Note: file can be /dev/null or \"\"")
		print("")
		print("If the file does not exist it creates it and")
		print("write the current time information.")
		print("You can add optional comments that will be added")
		print("to the end of the line.")
		s="%17.6f %s %10.3f %10.3f" % (secs,date,delta,delta2)
		print("")
		print("tymer can be called as a function:")
		print("from tymer import *")
		print('tymer()                          prints to stdout')                      
		print('tymer("file")                    prints to file')
		print('tymer("-i")                      use an internal file for saving time')
		print('tymer(["file","comments"])       prints to file with comments')
		print('tymer(["","comments"])           prints to stdout with comments')
		print('tymer(["/dev/null","comments"])  prints to stdout with comments')
		print("")
		print(s)
		if doquit : 
			quit()	
	
	if(file == "-i"):
		lines=tymer_save.getvalue()
		if len(lines) > 0 :
			lines=lines.split("\n")
			lines.reverse()
			if(len(lines[0]) == 0): lines=lines[1:]
	# calculate a delta
			ll=lines[-1]
			try:
				last=ll.split()
				pre=float(last[0])
			except:
				pre=secs
			ll=lines[0]
			try:
				last=ll.split()
				post=float(last[0])
			except:
				post=secs
		else:
			pre=secs
			post=secs		
		delta=secs-pre
		delta2=secs-post
	# print secs,date,delta
		s="%17.6f %s %10.3f %10.3f %s" % (secs,date,delta,delta2,comment)
		print(s)
		tymer_save.write(s+"\n")
		return(None)		
	
	#two cases
	exists=os.path.exists(file)
	# file exists
	if exists :
	# open it for read
		with open(file, "r") as myfile:
			lines=myfile.readlines()
	# close it
			myfile.close()
	# get the last line
		if len(lines) > 0 :
	# calculate a delta
			ll=lines[-1]
			try:
				last=ll.split()
				pre=float(last[0])
			except:
				pre=secs
			ll=lines[0]
			try:
				last=ll.split()
				post=float(last[0])
			except:
				post=secs
		else:
			pre=secs
			post=secs
		
		delta=secs-pre
		delta2=secs-post
	# print secs,date,delta
		s="%17.6f %s %10.3f %10.3f %s" % (secs,date,delta,delta2,comment)
		print(s)
		with open(file, "a") as myfile:
			myfile.write(s+"\n")
			myfile.close()
		if doquit : 
			quit()

	# file does not exist
	else:
		s="%17.6f %s %10.3f %10.3f %s" % (secs,date,delta,delta2,comment)
		print(s)
	#open it
		with open(file, "w") as myfile:
			myfile.write(s+"\n")
			myfile.close()
		if doquit : 
			quit()
if __name__ == '__main__':
	doquit=True
	tymer(sys.argv)
else:
	from io import StringIO
	tymer_save = StringIO()


	
