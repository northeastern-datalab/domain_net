cd ../table-union-master/

echo 'Running TUS_sem_file.py...'
python TUS_sem_file.py
echo 'Finished running TUS_sem_file.py...'

cd ../TABLE_UNION_OUTPUT/

echo 'Creating annotation database...'
sqlite3 "annotation_og" "DROP TABLE all_labels" ".exit"
sqlite3 "annotation_og" ".mode csv" ".import all_labels.csv all_labels" ".exit"

echo 'Finished creating the annotation database'

echo 'Running the TUS pipeline...'
cd ../table-union-master/
./run.sh
echo 'Finished running the TUS pipeline'