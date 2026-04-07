set -e -o pipefail

go_train() {
    local now=$(date +"%Y-%m-%d_%H-%M-%S")
    local base_dir="${TRAIN_DIR:-${PWD}/run}"
    local dir="${base_dir}/training-${now}"
    mkdir -p "${dir}/images" || return ${?}
    ln -sfn "training-${now}" "${base_dir}/last"

    echo "output will be in ${dir}"
    cd "${dir}" || return ${?}
}

fullpath() {
    for f in "$@"; do
        readlink -f "$f"
    done
}

parse_training_args() {
    local log=${1} ; shift
    if [ ! -f ${log} ] ; then
        return 0
    fi
    grep "${expression}" ${log} | sed 's,[^{]*{,,;s,}[^}]*$,,;s,[ "],,g;s/:,/:"",/g;s/,/ --/g;s,:, ,g;s,^,--,;s,$, ,;s, true, ,;s, false , ,;s, *$,,'
}
