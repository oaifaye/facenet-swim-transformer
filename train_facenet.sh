for i in $(ps -ax |grep train_facenet |awk '{print $1}')
do
 id=`echo $i |awk -F"/" '{print $1}'`
 kill -9  $id
done

nohup python -u train_facenet.py > train_facenet.log &
tail -f train_facenet.log
