totay=$(date "+%Y-%m-%d")
dst_dir=${today}
if [ ! -d $dst_dir ];then
    mkdir $dst_dir
else
    rm -rf $dst_dir
    mkdir $dst_dir
fi

sh start.sh
sleep 4h
sh stop.sh
sleep 1m

python3 send_email.py
