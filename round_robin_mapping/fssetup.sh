#!/bin/sh
ROUND_ROBIN_RANGE = 63  # i.e. the number of IONs
NET_START=2

LUSTRE_CONF=/etc/modprobe.d/lustre.conf

# ion.conf file in each ION
# contains the information of all IONs
# Format:  
#   ION1_IP  ION1_HOSTNAME  ION1  1
#   ION2_IP  ION2_HOSTNAME  ION2  2
#   ...
ION_CONF=/etc/ion.conf

mds0_ip="X.X.X.7"   # modify to your IP of Lustre MDS
mds1_ip="X.X.X.8"   # modify to your IP of Lustre MDS

HN=$(hostname)

function umount_fs()
{
	umount  /LustreDir
	lctl network down
	lustre_rmmod
}

function kernel_param()
{
    # example kernel params
    # need add more your own Lustre's params
    sysctl -w net.core.rmem_max=67108864
    sysctl -w net.core.wmem_max=67108864
}

function ion_router()
{
   net=$(cat $ION_CONF|awk '{if ($2 == hn) print $4}' hn=$HN)
   echo "options lnet networks=tcp1,tcp$net forwarding=enabled" > $LUSTRE_CONF
   echo "options ksocklnd sock_timeout=100 peer_credits=16 credits=64" >> $LUSTRE_CONF
   modprobe lnet
   lctl network configure
}

function cn_mount()
{
   ip_addr=$1
   fs_name=$2
   mount_point=$3

   mkdir -p $mount_point
   mount -t lustre -o localflock,nosuid ${ip_addr}@tcp1:/$fs_name $mount_point
}


function cn_router()
{
   NODE=$(hostname|cut -c 3-)
   net=$((NODE%ROUND_ROBIN_RANGE+NET_START))
   echo "options lnet networks=tcp$net dead_router_check_interval=60 router_ping_timeout=10 check_routers_before_use=1 avoid_asym_router_failure=1" >$LUSTRE_CONF
   echo "options ksocklnd sock_timeout=100 peer_credits=8 credits=64" >>$LUSTRE_CONF

   modprobe ptlrpc
   for ip in `awk '{if ($4 == net) print $1}' net=$net $ION_CONF`
   do
     lctl --net tcp1 add_route ${ip}@tcp$net
   done
   modprobe lustre
   echo -n "Mount FS with ION support ... "
   sleep 5
   chmod 1777 /tmp
   echo "OK!"
}

function cn_chkroute()
{
   res=`lctl show_route|grep down`
   if [ "$res" != "" ]
     then
     sleep 10
     lctl ping ${mds0_ip}@tcp1 > /tmp/mount.log
     sleep 50
   fi
}

####main####

if [[ "$HN" == cn* ]]; then
	if [ "$1" = "umountfs" ];then
		umount_fs 
	fi
	kernel_param
	cn_router
	cn_chkroute
	cn_mount $mds0_ip FS /LustreDir

elif [[ "$HN" == ion* ]]; then
	if [ "$1" = "umountfs" ];then
		umount_fs 
	fi
	kernel_param
	ion_router	

else
  echo "Wrong Hostname!"
fi
