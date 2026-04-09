set -e -o pipefail

# Set pyenv version from the repo's .python-version file.
# BASE must be set to the bin/ directory before sourcing this file.
export PYENV_VERSION=$(cat "${BASE}/../.python-version" 2>/dev/null || echo "")

# source_conf CONF [VAR ...]
#
# Sources CONF while preserving any VARs that were set in the calling
# environment (e.g. OUTER_STEPS=5555 ./bin/run-meta-train-grid).
# Script-local defaults are still overridden by CONF as usual — only
# genuine env vars (passed by the caller) are protected.
source_conf() {
    local _conf=$1; shift
    [ -f "${_conf}" ] || return 0

    local -a _keys _vals
    local _v _val
    for _v in "$@"; do
        if _val=$(printenv "$_v" 2>/dev/null); then
            _keys+=("$_v")
            _vals+=("$_val")
        fi
    done

    source "${_conf}"

    local _i
    for _i in "${!_keys[@]}"; do
        printf -v "${_keys[$_i]}" '%s' "${_vals[$_i]}"
    done
}

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
