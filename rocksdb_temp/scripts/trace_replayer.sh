../cmake-build-debug/db_bench --benchmarks=replay --use_existing_db=true --db="/tmp/block_cache_trace" \
--trace_file=/tmp/op_trace_file --num_column_families=5 --use_direct_reads=false \
--txt_file=../trace_data_dir/read/trace-human_readable_trace.txt

