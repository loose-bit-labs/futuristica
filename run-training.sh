#!/bin/bash	

set -e -o pipefail

_run_training_main() {
	########################################################################################

	local file=${1-images/lenna.png}

	# how long to run
	local training=22

	# this is the image size, it will scale up or down too
	local size=512

	# other options: rgb and yuv, ycbcr seem to work best "in general"
 	local colorspace="ycbcr"

	# this seems to be best but you can try the others
	# these seem to work best: mse l1 huber bce klDiv
 	local loss_fn="mse"

	# recommend using 16 (wide), 
	# more than 4 layers and the glsl can be heavy on mobile, etc
	local model_size=16
	local model_count=4

	# this will allow you to reuse the output of a prior run
	local ckp="futuristica.npz"

	# use this if you want to start fresh
 	local ckp="" 

	# no longer suggest changing this... 3 is dialed in now to 
	# encode vec2 (2d) to mat4 (16d) but go nuts!

	local coding=3 

	########################################################################################
	
	file=$(fullpath ${file})
	if [ ! -f ${file} ] ; then
		echo "where is this ${file} of which you speak?"
		return  1
	fi

	if [ "" != "${ckp}" ] ; then
		ckp=$(fullpath ${ckp})
		if [ ! -f "${ckp}" ] ; then
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

    time ../../futuristica.py          \
		--weights     weights.npz       \
		--generated   images/output.png \
		--loss_fn     ${loss_fn}   	   \
		--coding      ${coding}         \
		--training    ${training}       \
		--size        ${size}           \
		--image       ${file}           \
		--colorspace  ${colorspace}     \
		--ckp         "${ckp}"          \
		--model_size  ${model_size}    \
		--model_count ${model_count}   \
		2>&1 | tee train.log || return ${?}
	echo "Training completed"

	# note: this -s only works of model_size 16, so change this if you change that... 
	../../translate.py -s --coding ${coding} --colorspace ${colorspace} weights.npz > output.glsl

	ls -lathr ${dir}/*.*

	echo "output is in ${PWD}"

	cp -i weights.npz ../../zed.npz

	########################################################################################
}

fullpath () { 
    local f;
    for f in $*; do
        ( ( cd ${f} 2> /dev/null && echo ${PWD} ) || ( cd $(dirname ${f} ) 2> /dev/null && echo ${PWD}/$( basename ${f} ) ) );
    done
}

_run_training_main ${*}
