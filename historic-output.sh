#!/bin/bash	

_historic_output_main() {
	local dir=${1-last}

	local mp4=${dir}/training.mp4

	local nu=0
	if [ -f ${mp4} ] ; then
		local last=$(ls -tr ${dir}/images/output-*png | tail -1)
		xv ${last} &
		if [ ${last} -nt ${mp4} ] ; then
			echo newer png file
			nu=1
			mv -f ${mp4} /tmp/bak-training-$(date +%s).mp4
		else
			echo ho-hum
		fi
	else
		nu=1
	fi

	if [ 1 = ${nu} ] ; then
		ls -tr ${dir}/images/output-*png \
		| xargs cat \
		| ffmpeg -loop 0 -f image2pipe -framerate 24 -i - ${mp4}  \
		|| return ${?}
	fi

	mplayer -loop 0 -fixed-vo ${dir}/training.mp4
}

_historic_output_main ${*}
