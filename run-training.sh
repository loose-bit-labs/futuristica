#!/bin/bash	

set -e -o pipefail

_run_training_main() {
	########################################################################################

	local file=${1-images/lenna.png}

	# translates into 1 minute each
	local training=10
	local coding=3
training=1

	# this will allow you to reuse the output of a prior run
	# be sure to use the abs
	local ckp="futuristica.npz"

	########################################################################################

	file=$(fullpath ${file})
	if [ ! -f ${file} ] ; then
		echo "where is this ${file} of which you speak?"
		return  1
	fi

	if [ "" != "${ckp}"] ; then
		ckp=$(fullpath ${ckp})
		if [ ! -f ${ckp} ] ; then
			echo "where is this ${ckp} of which you speak?"
			return  1
		fi
	fi

	########################################################################################

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

	########################################################################################

	echo "Starting training"

    time ../../futuristica.py \
		--weights   weights.npz \
		--generated images/output.png \
		--coding    ${coding} \
		--training  ${training} \
		--image     ${file} \
		--ckp       ${ckp} \
		2>&1 | tee train.log || return ${?}
	echo "Training completed"

	../../translate.py --coding ${coding} weights.npz > output.glsl

	ls -lathr ${dir}/*.*

	xv ${file} &
	xv images/output.png &

	echo "output is in ${PWD}"

	########################################################################################
}

fullpath () { 
    local f;
    for f in $*; do
        ( ( cd ${f} 2> /dev/null && echo ${PWD} ) || ( cd $(dirname ${f} ) 2> /dev/null && echo ${PWD}/$( basename ${f} ) ) );
    done
}

_run_training_main ${*}
