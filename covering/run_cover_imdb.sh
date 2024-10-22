for i in 0 1 2 3 4; 
do
    python main_cover.py --layers 5 --hidden 128 --ind-type None --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 5 --hidden 128 --ind-type RNI  --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 5 --hidden 128 --ind-type RP  --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 5 --hidden 128 --ind-type Tinhofer --k-weak 0 --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 5 --hidden 128 --ind-type Tinhofer --k-weak 9 --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 5 --hidden 128 --ind-type LPE --dataset IMDB-BINARY --seed $i

    python main_cover.py --layers 2 --hidden 128 --ind-type None --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 2 --hidden 128 --ind-type RNI  --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 2 --hidden 128 --ind-type RP  --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 2 --hidden 128 --ind-type Tinhofer --k-weak 0 --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 2 --hidden 128 --ind-type Tinhofer --k-weak 9 --dataset IMDB-BINARY --seed $i
    python main_cover.py --layers 2 --hidden 128 --ind-type LPE --dataset IMDB-BINARY --seed $i
    echo a
done

python cover.py --ind-type None --layers 5 --dataset IMDB-BINARY
python cover.py --ind-type RP --layers 5 --dataset IMDB-BINARY
python cover.py --ind-type RNI --layers 5 --dataset IMDB-BINARY
python cover.py --ind-type Tinhofer --k-weak 0 --layers 5 --dataset IMDB-BINARY
python cover.py --ind-type Tinhofer --k-weak 9 --layers 5 --dataset IMDB-BINARY
python cover.py --ind-type LPE --layers 5 --dataset IMDB-BINARY