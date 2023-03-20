BLOCK_SIZE = 8
SIZE = 512
C=2
#### ICL-NUIM TRAJECTORIES PARAMETERS  ####
0 = -s 5.0 -p 0.34,0.5,0.24 	-z 1 -c $C -r 1 -k 481.2,-480,320,240 -B $(BLOCK_SIZE) -v $(SIZE) -m 0.1 -F
1 = -s 5.0 -p 0.485,0.5,0.55 	-z 1 -c $C -r 1 -k 481.2,-480,320,240 -B $(BLOCK_SIZE) -v $(SIZE) -m 0.1 -F  
2 = -s 5.0 -p 0.34,0.5,0.24 	-z 1 -c $C -r 1 -k 481.2,-480,320,240 -B $(BLOCK_SIZE) -v $(SIZE) -m 0.1 -F
3 = -s 5.0 -p 0.2685,0.5,0.4 	-z 1 -c $C -r 1 -k 481.2,-480,320,240 -B $(BLOCK_SIZE) -v $(SIZE)  -m 0.1 -F 
live = -s 5.0 -p 0.5,0.5,0.5 -z 1 -c 1 -r 1 -k 580.8,581.8,308.8,253.0 -B $(BLOCK_SIZE) -v $(SIZE) -m 0.1
fr2desk = -s 5.0 -p 0.5,0.5,0.2 -z 1 -c 2 -r 1 -k 580.8,581.8,308.8,253.0 -B $(BLOCK_SIZE) -v $(SIZE) -m 0.1 -F
# fr2xyz = 	-s 8.0 -p 0.5,0.5,0.5	-z 1 -c 2 -r 1 -k 580.8,581.8,308.8,253.0 -B $(BLOCK_SIZE) -v $(SIZE),$(SIZE),$(SIZE)  -m 0.1 -F 

ROOT_DIR=$(shell pwd)
TOON_DIR=${ROOT_DIR}/TooN/install_dir
TOON_INCLUDE_DIR=${TOON_DIR}/include/


all : TooN
	mkdir -p build/
	cd build/ && cmake -DCMAKE_BUILD_TYPE=Release -DTOON_INCLUDE_PATH=${TOON_INCLUDE_DIR} $(CMAKE_ARGUMENTS) ..
	$(MAKE) -C build  $(MFLAGS) $(SPECIFIC_TARGET)


debug: TooN
	mkdir -p build/
	mkdir -p build/logs/
	cd build/ && cmake -DCMAKE_BUILD_TYPE=Debug -DTOON_INCLUDE_PATH=${TOON_INCLUDE_DIR} $(CMAKE_ARGUMENTS) ..
	$(MAKE) -C build $(MFLAGS)


TooN:
	git clone https://github.com/edrosten/TooN.git
	cd TooN &&  git checkout 92241416d2a4874fd2334e08a5d417dfea6a1a3f
	mkdir -p ${TOON_DIR}
	cd TooN && ./configure --prefix=${TOON_DIR} && make install

#### Usage ####
demo_cup_bottle:
	@demo/cup_and_bottle.sh

demo_carrot: 
	@demo/carrot.sh

demo_multi_obj: 
	@demo/multi_moving.sh

demo_rotate_book: 
	@demo/rotate_book.sh

demo_syn_2objs:
	@demo/syn_2objs.sh

#### GENERAL GENERATION ####
clean :
	rm -rf build
	rm -fr TooN

.PHONY : clean debug all 