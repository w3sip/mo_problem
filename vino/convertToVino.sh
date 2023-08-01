SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

MODELDIR=${SCRIPT_DIR}/../model
TEMPDIR=${SCRIPT_DIR}/../model-tmp
OUTDIR=${SCRIPT_DIR}/../model-out
DATATYPE=FP16

export PATH=$PATH:$SCRIPT_DIR/../bin

function checkRet()
{
    retVal=$1
    msg=$2
    if [ $retVal -ne 0 ]; then
        echo $msg
        exit -1
    fi
}


rm -rf $TEMPDIR
mkdir -p $TEMPDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR

MODIR=/usr/local/lib/python3.6/dist-packages/openvino/tools/mo

EXTDIR=$MODELDIR/../vino/ext_modules

# if [[ $DATATYPE == FP16 ]]; then
#     FP16_VAL="--compress_to_fp16"
# fi


COMMON_OPTIONS=""

PROTODIR=$MODIR/front/caffe/proto
PATCHED_PROTODIR=$TEMPDIR/proto

mkdir -p $PATCHED_PROTODIR
${PYTHONEXE} $PROTODIR/generate_caffe_pb2.py \
    --input_proto ${EXTDIR}/mo_caffe.proto \
    --output $PATCHED_PROTODIR
checkRet $? "Failed to compile protobuf"

# Patch Vino installation
cp $PATCHED_PROTODIR/caffe_pb2.py $PROTODIR
cp $EXTDIR/mo_caffe.proto $PROTODIR
cp $EXTDIR/yolov3_detection_output_ext.py $PROTODIR/..
cp $EXTDIR/yolov3detectionoutput.py $PROTODIR/..
cp $EXTDIR/CustomLayersMapping.xml $PROTODIR/..


# YoloV3 models...
echo "Converting YoloV3, extensions are in $EXTDIR :"
ls -la $EXTDIR
export PYTHONPATH=$PYTHONPATH:$EXTDIR
mo --input_model "${MODELDIR}/model.caffemodel" \
        -d "${MODELDIR}/deploy.prototxt" \
        -s '255.0' \
        --output_dir ${OUTDIR} \
        --log_level=DEBUG ${FP16_VAL} \
        --data_type FP16 \
        --use_legacy_frontend \
        --input_shape [1,3,416,416] 2>&1

        # --extensions $EXTDIR \
        # --caffe_parser_path $PROTODIR/caffe_pb2.py \

checkRet $? "Failed to convert the model at ${MODELDIR}"


