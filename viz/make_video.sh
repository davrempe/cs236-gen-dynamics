#!/bin/bash
# Check for input
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "[ USAGE ]:" $0 "<path_to_dir_name> <file_type> <output_fps> <output_filename> [<overwrite_without_asking = 0>]"
    exit
fi

INPUTDIR=${1%/}
IMTYPE=$2
FPS=$3
OUTPUTFNAME=${4}
NOASK=0
if [ $# -eq 5 ] && [ ${5} -gt 0 ]; then
    NOASK=1
fi

# Now merge them. Using default good codecs
#mencoder "mf://${INPUTDIR}/*.${IMTYPE}" -mf fps=${INPUTFPS}:type=${IMTYPE} -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell:vbitrate=7000 -oac copy -speed 0.1 -o ${OUTPUTFNAME}

if [[ "${OUTPUTFNAME}" = /* ]]; then
	FINALOUTFNAME="${OUTPUTFNAME}"
else
	FINALOUTFNAME="`pwd`/${OUTPUTFNAME}"
fi

echo ${FINALOUTFNAME}

cd ${INPUTDIR}

if [ ${NOASK} -eq 0 ]; then
    ffmpeg -framerate ${FPS} -pattern_type glob -i "*.${IMTYPE}" -c:v libx264 -vf format=yuv420p -b 10000k ${FINALOUTFNAME}
else
    ffmpeg -y -framerate ${FPS} -pattern_type glob -i "*.${IMTYPE}" -c:v libx264 -vf format=yuv420p -b 10000k ${FINALOUTFNAME} 
fi

cd -
