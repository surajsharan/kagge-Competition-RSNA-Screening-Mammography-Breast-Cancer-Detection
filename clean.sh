# Get a list of all python processes
pids=$(ps -ef | grep '[p]ython' | awk '{print $2}')

# Kill each process in the list
for pid in $pids
do
  kill $pid
done