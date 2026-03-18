#!/bin/bash
set -e
set -x

FS_PATH=$1
SUBJECT=$(basename "$FS_PATH")
SUBJECTS_DIR=$(dirname "$FS_PATH")

source /usr/local/freesurfer/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$SUBJECTS_DIR

mkdir -p $FS_PATH/bem/watershed

mri_watershed -atlas -useSRAS \
  -surf $FS_PATH/bem/watershed/$SUBJECT \
  $FS_PATH/mri/T1.mgz \
  $FS_PATH/bem/watershed/ws

cp $FS_PATH/bem/watershed/${SUBJECT}_inner_skull_surface $FS_PATH/bem/inner_skull.surf
cp $FS_PATH/bem/watershed/${SUBJECT}_outer_skull_surface $FS_PATH/bem/outer_skull.surf
cp $FS_PATH/bem/watershed/${SUBJECT}_outer_skin_surface  $FS_PATH/bem/outer_skin.surf

echo "Surfaces created."
