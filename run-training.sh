#!/bin/bash	

set -e -o pipefail

_run_training_main() {
	local file=${1-images/lenna.png}

	# this is roughly 1 minute per
	local training=60

	file=$(fullpath ${file})
	if [ ! -f ${file} ] ; then
		echo "where is this ${file} of which you speak?"
		return  1
	fi

	local now=$(date +"%Y-%m-%d_%H-%M-%S")
	local dir=${PWD}/"run/training-${now}"
	mkdir -p ${dir}/images
	rm -f last
	ln -s ${dir} last

	echo "output will be in ${dir}"

	echo "Backing up model"
	awk '/class MLP/ {ON=1} /def forward/ {ON=0} ON{print}' futuristica.py > .m
	mv -i .m ${dir}/model.py

	cd ${dir}

	echo "Starting training"

    time ../../futuristica.py \
		--weights   weights.npz \
		--generated images/output.png \
		--training  ${training} \
		--image     ${file} \
		2>&1 | tee train.log || return ${?}

	echo "Training completed"

	ls -lathr ${dir}/*.*

	xv ${file} &
	xv images/output.png &

	echo "output is in ${PWD}"
}

fullpath () { 
    local f;
    for f in $*; do
        ( ( cd ${f} 2> /dev/null && echo ${PWD} ) || ( cd $(dirname ${f} ) 2> /dev/null && echo ${PWD}/$( basename ${f} ) ) );
    done
}

_run_training_main ${*}
