#!/bin/bash
set -e
set -x

FS_PATH=$1
SUBJECT=$(basename "$FS_PATH")
SUBJECTS_DIR=$(dirname "$FS_PATH")

export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$SUBJECTS_DIR

mkdir -p $FS_PATH/bem/watershed

mri_watershed -atlas -useSRAS \
  -surf $FS_PATH/bem/watershed/$SUBJECT \
  $FS_PATH/mri/T1.mgz \
  $FS_PATH/bem/watershed/ws

cp $FS_PATH/bem/watershed/${SUBJECT}_inner_skull_surface $FS_PATH/bem/inner_skull.surf
cp $FS_PATH/bem/watershed/${SUBJECT}_outer_skull_surface $FS_PATH/bem/outer_skull.surf
cp $FS_PATH/bem/watershed/${SUBJECT}_outer_skin_surface  $FS_PATH/bem/outer_skin.surf

echo "Watershed done. Checking topology with mris_info..."
mris_info $FS_PATH/bem/inner_skull.surf 2>&1 | grep -E "vertices|faces"
mris_info $FS_PATH/bem/outer_skull.surf 2>&1 | grep -E "vertices|faces"
mris_info $FS_PATH/bem/outer_skin.surf  2>&1 | grep -E "vertices|faces"

echo '{"brainlife": [{"type": "info", "msg": "Watershed BEM surfaces created with FreeSurfer 7.1.1. Check logs for topology."}]}' > product.json
