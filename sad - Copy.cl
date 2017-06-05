
// TODO: Add OpenCL kernel code here.

__kernel void copyToLocal()
{
	printf("useless kernel\n"); 
}

__kernel void sad(
	__global int* Rmat,
	__global int* Lmat,
	__global int* SADmat,
	__global int* imdim,
	__global int* paddeddim,
	__global bool* first,
	__local int *Rmatloc,
	__local int *Lmatloc,
	int local_size,
	__global bool* local_bool,
	__global int* localItemSize,
	__global bool* unroll,
	__global int* dispLevel,
	__global int* rad,
	__global bool* last
	)
{

/*
Each kernel will calculate the disparity for a pixel
Local size will be as large as there are pixels
NDRange will be 2 for h and w in pixel matrix
Total of 4 WG, and GS = total_im_size/4
*/

	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int ga = get_group_id(0);
	int gb = get_group_id(1);
	int gc = get_local_id(0);
	int gd = get_local_id(1);

	//if (gx < 5 && gy < 5){ 
		//printf("global(%d, %d)| group(%d, %d) | local(%d, %d)\n", gx, gy, ga, gb, gc, gd);
	//}

	/*for (int i=0; i < local_size; i++) {
		if( gx < 5 && gy < 5 && i < 3)
			printf("global(%d, %d)| group(%d, %d) | local(%d, %d) | Rmat[%d]: %d\n", 
				gx, gy, ga, gb, gc, gd, i, Rmat[i]);
		//if( gx == 0 && gy == 0)
			//p[i] = i;
	}*/

	__private int LOCAL_HEIGHT = 5;
	__private int LOCAL_WIDTH = 512;

	if(gx < rad[0] || gx > imdim[1] - dispLevel[0] - rad[0] || 
		gy < rad[0] || gy >= imdim[0] - rad[0]) return;
	else{

		__private int i, j, m;
		__private int leftBound, rightBound, topBound;
		__private int numGroupsX = paddeddim[1]/localItemSize[1];
				
		// load local memory
		if(local_bool[0]){

			// load local memory in parallel
			if(local_bool[1]) {
				// determine boundaries of work-item IDs in workgroup
				if (ga == 0) // first workgroup
					leftBound = rad[0];
				else
					leftBound = ga * localItemSize[1];
				
				if (ga == (int)((imdim[1] - dispLevel[0] - rad[0])/localItemSize[1])) // last workgroup
					rightBound = imdim[1] - dispLevel[0] - rad[0];
				else
					rightBound = (ga + 1) * localItemSize[1] - 1;

				if ((localItemSize[1] < rad[0] && gb < rad[0]) || gb == 0)
					topBound = rad[0];
				else
					topBound = gb * localItemSize[0];

				// populate local memory

				if(gy == topBound){ 

					// left and right matrices within workgroup boundaries
					Rmatloc[0 * LOCAL_WIDTH + gx] = Rmat[(gy - 2) * paddeddim[1] + gx];
					Rmatloc[1 * LOCAL_WIDTH + gx] = Rmat[(gy - 1) * paddeddim[1] + gx];
					Rmatloc[2 * LOCAL_WIDTH + gx] = Rmat[(gy - 0) * paddeddim[1] + gx];
					Rmatloc[3 * LOCAL_WIDTH + gx] = Rmat[(gy + 1) * paddeddim[1] + gx];
					Rmatloc[4 * LOCAL_WIDTH + gx] = Rmat[(gy + 2) * paddeddim[1] + gx];	
					Lmatloc[0 * LOCAL_WIDTH + gx] = Lmat[(gy - 2) * paddeddim[1] + gx];
					Lmatloc[1 * LOCAL_WIDTH + gx] = Lmat[(gy - 1) * paddeddim[1] + gx];
					Lmatloc[2 * LOCAL_WIDTH + gx] = Lmat[(gy - 0) * paddeddim[1] + gx];
					Lmatloc[3 * LOCAL_WIDTH + gx] = Lmat[(gy + 1) * paddeddim[1] + gx];
					Lmatloc[4 * LOCAL_WIDTH + gx] = Lmat[(gy + 2) * paddeddim[1] + gx];

					// boundary
					if(gx == leftBound){ 
						Rmatloc[0 * LOCAL_WIDTH + gx - 1] = Rmat[(gy - 2) * paddeddim[1] + gx - 1];
						Rmatloc[1 * LOCAL_WIDTH + gx - 1] = Rmat[(gy - 1) * paddeddim[1] + gx - 1];
						Rmatloc[2 * LOCAL_WIDTH + gx - 1] = Rmat[(gy - 0) * paddeddim[1] + gx - 1];
						Rmatloc[3 * LOCAL_WIDTH + gx - 1] = Rmat[(gy + 1) * paddeddim[1] + gx - 1];
						Rmatloc[4 * LOCAL_WIDTH + gx - 1] = Rmat[(gy + 2) * paddeddim[1] + gx - 1];
						Rmatloc[0 * LOCAL_WIDTH + gx - 2] = Rmat[(gy - 2) * paddeddim[1] + gx - 2];
						Rmatloc[1 * LOCAL_WIDTH + gx - 2] = Rmat[(gy - 1) * paddeddim[1] + gx - 2];
						Rmatloc[2 * LOCAL_WIDTH + gx - 2] = Rmat[(gy - 0) * paddeddim[1] + gx - 2];
						Rmatloc[3 * LOCAL_WIDTH + gx - 2] = Rmat[(gy + 1) * paddeddim[1] + gx - 2];
						Rmatloc[4 * LOCAL_WIDTH + gx - 2] = Rmat[(gy + 2) * paddeddim[1] + gx - 2];
						Lmatloc[0 * LOCAL_WIDTH + gx - 1] = Lmat[(gy - 2) * paddeddim[1] + gx - 1];
						Lmatloc[1 * LOCAL_WIDTH + gx - 1] = Lmat[(gy - 1) * paddeddim[1] + gx - 1];
						Lmatloc[2 * LOCAL_WIDTH + gx - 1] = Lmat[(gy - 0) * paddeddim[1] + gx - 1];
						Lmatloc[3 * LOCAL_WIDTH + gx - 1] = Lmat[(gy + 1) * paddeddim[1] + gx - 1];
						Lmatloc[4 * LOCAL_WIDTH + gx - 1] = Lmat[(gy + 2) * paddeddim[1] + gx - 1];
						Lmatloc[0 * LOCAL_WIDTH + gx - 2] = Lmat[(gy - 2) * paddeddim[1] + gx - 2];
						Lmatloc[1 * LOCAL_WIDTH + gx - 2] = Lmat[(gy - 1) * paddeddim[1] + gx - 2];
						Lmatloc[2 * LOCAL_WIDTH + gx - 2] = Lmat[(gy - 0) * paddeddim[1] + gx - 2];
						Lmatloc[3 * LOCAL_WIDTH + gx - 2] = Lmat[(gy + 1) * paddeddim[1] + gx - 2];
						Lmatloc[4 * LOCAL_WIDTH + gx - 2] = Lmat[(gy + 2) * paddeddim[1] + gx - 2];
					}
					if(gx == rightBound) { 
						Rmatloc[0 * LOCAL_WIDTH + gx + 1] = Rmat[(gy - 2) * paddeddim[1] + gx + 1];
						Rmatloc[1 * LOCAL_WIDTH + gx + 1] = Rmat[(gy - 1) * paddeddim[1] + gx + 1];
						Rmatloc[2 * LOCAL_WIDTH + gx + 1] = Rmat[(gy - 0) * paddeddim[1] + gx + 1];
						Rmatloc[3 * LOCAL_WIDTH + gx + 1] = Rmat[(gy + 1) * paddeddim[1] + gx + 1];
						Rmatloc[4 * LOCAL_WIDTH + gx + 1] = Rmat[(gy + 2) * paddeddim[1] + gx + 1];
						Rmatloc[0 * LOCAL_WIDTH + gx + 2] = Rmat[(gy - 2) * paddeddim[1] + gx + 2];
						Rmatloc[1 * LOCAL_WIDTH + gx + 2] = Rmat[(gy - 1) * paddeddim[1] + gx + 2];
						Rmatloc[2 * LOCAL_WIDTH + gx + 2] = Rmat[(gy - 0) * paddeddim[1] + gx + 2];
						Rmatloc[3 * LOCAL_WIDTH + gx + 2] = Rmat[(gy + 1) * paddeddim[1] + gx + 2];
						Rmatloc[4 * LOCAL_WIDTH + gx + 2] = Rmat[(gy + 2) * paddeddim[1] + gx + 2];
						for(i = rightBound + 1; i < rightBound + dispLevel[0] + rad[0]; i++) {
							Lmatloc[0 * LOCAL_WIDTH + i] = Lmat[(gy - 2) * paddeddim[1] + i];
							Lmatloc[1 * LOCAL_WIDTH + i] = Lmat[(gy - 1) * paddeddim[1] + i];
							Lmatloc[2 * LOCAL_WIDTH + i] = Lmat[(gy - 0) * paddeddim[1] + i];
							Lmatloc[3 * LOCAL_WIDTH + i] = Lmat[(gy + 1) * paddeddim[1] + i];
							Lmatloc[4 * LOCAL_WIDTH + i] = Lmat[(gy + 2) * paddeddim[1] + i];
						}
					}
				}
				else { 
					while(!last[(gy - 1) * numGroupsX + ga]) {
						//printf("waiting on last[%d, %d]: %d\n", gy - 1, ga, last[(gy - 1) * numGroupsX + ga]);
						printf("");
					}
					Rmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx] = Rmat[(gy + rad[0]) * paddeddim[1] + gx];
					Lmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx] = Lmat[(gy + rad[0]) * paddeddim[1] + gx];
					if(gx == leftBound){ 
						Rmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx - 2] = Rmat[(gy + rad[0]) * paddeddim[1] + gx - 2];
						Rmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx - 1] = Rmat[(gy + rad[0]) * paddeddim[1] + gx - 1];
						Lmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx - 2] = Lmat[(gy + rad[0]) * paddeddim[1] + gx - 2];		
						Lmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx - 1] = Lmat[(gy + rad[0]) * paddeddim[1] + gx - 1];		
					}	
					if(gx == rightBound){ 
						Rmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx + 1] = Rmat[(gy + rad[0]) * paddeddim[1] + gx + 1];
						Rmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + gx + 2] = Rmat[(gy + rad[0]) * paddeddim[1] + gx + 2];
						for(i = rightBound + 1; i < rightBound + dispLevel[0] + rad[0]; i++) 
							Lmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + i] = Lmat[(gy + rad[0]) * paddeddim[1] + i];
					}			
				}
			}
			// load local memory (by first work-item)
			else if(!last[(gy - 1) * numGroupsX + ga]) { // check if local mem copied yet
				// determine boundaries of local memory copied
				if (ga == 0)
					leftBound = 0;
				else
					leftBound = ga * localItemSize[1] - rad[0];

				if (ga == (int)((imdim[1] - dispLevel[0] - rad[0])/localItemSize[1])) // last workgroup
					rightBound = imdim[1] - dispLevel[0] + 1;
				else
					rightBound = (ga + 1) * localItemSize[1] + rad[0];

				// top boundary is work-item ID
				/*if(localItemSize[1] == 1 && gb == 2)
					topBound = rad[0];
				else if (localItemSize[1] == 2 && gb == 1)
					topBound = rad[0];
				else if (gb == 0)
					topBound = rad[0];*/
				if ((localItemSize[1] < rad[0] && gb < rad[0]) || gb == 0)
					topBound = rad[0];
				else
					topBound = gb * localItemSize[0];
	
				// populate local memory 

				if(gy == topBound)
					for (i = 0; i < LOCAL_HEIGHT; i++) {
						for (j = leftBound; j < rightBound; j++)
							Rmatloc[i * LOCAL_WIDTH + j] = Rmat[(gy - rad[0] + i) * paddeddim[1] + j];
						for (m = leftBound; m < rightBound + dispLevel[0]; m++)
							Lmatloc[i * LOCAL_WIDTH + m] = Lmat[(gy - rad[0] + i) * paddeddim[1] + m];
					}
				else{ 
					//printf("last[%d, %d]: %d\n", gy - 1, ga, last[(gy - 1) * numGroupsX + ga]);
					while(!last[(gy - 1) * numGroupsX + ga]) {
						printf("");
					}
					for (j = leftBound; j < rightBound; j++)
						Rmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + j] = Rmat[(gy + rad[0]) * paddeddim[1] + j];
					for (m = leftBound; m < rightBound + dispLevel[0]; m++)
						Lmatloc[((gy - topBound - 1) % LOCAL_HEIGHT) * LOCAL_WIDTH + m] = Lmat[(gy + rad[0]) * paddeddim[1] + m];
					}
					
				first[gy * numGroupsX + ga] = true;
			}
			//printf("local memory loaded: global(%d) | group(%d) | local(%d)\n", gx, ga, gc);
		}

		// synchronize workgroup when loading in parallel
		//if(local_bool[0] && local_bool[1])
			//barrier(CLK_LOCAL_MEM_FENCE);

		__private int k, h, w, winsum;
		__private int temp, width, wink, winlength;
	
		width = LOCAL_WIDTH;//paddeddim[1];

		winlength = 2 * rad[0] + 1;
		temp = 255 * winlength * winlength;
		wink = 0;

		for(k = 0; k < dispLevel[0]; k++){
			winsum = 0;

			// loop for SAD calculation (local and global implementations)
			if(!unroll[0]){
				if(local_bool[0])
					for(h = 0; h < LOCAL_HEIGHT; h++){ 
						for(w = gx - rad[0]; w < gx + rad[0] + 1; w++){
							winsum += abs(Rmatloc[h * LOCAL_WIDTH + w] - Lmatloc[h * LOCAL_WIDTH + w + k]);
						}
					}
				else
					for(h = gy - rad[0]; h < gy + rad[0] + 1; h++){ 
						for(w = gx - rad[0]; w < gx + rad[0] + 1; w++){
							winsum += abs(Rmat[h * paddeddim[1] + w] - Lmat[h * paddeddim[1] + w + k]);
						}
					}
			}
			// unrolled loop for local implementation
			else if(local_bool[0]){ 
				winsum = abs(Rmatloc[0 * width + gx -2] -	Lmatloc[0 * width + gx -2 + k]) + 
					abs(Rmatloc[0 * width + gx -1] -	Lmatloc[0 * width + gx -1 + k]) +
					abs(Rmatloc[0 * width + gx] -		Lmatloc[0 * width + gx + k]) + 
					abs(Rmatloc[0 * width + gx + 1] - Lmatloc[0 * width + gx + 1 + k]) + 
					abs(Rmatloc[0 * width + gx + 2] - Lmatloc[0 * width + gx + 2 + k]);
				winsum += abs(Rmatloc[1 * width + gx -2] -	Lmatloc[1 * width + gx -2 + k]) + 
						abs(Rmatloc[1 * width + gx -1] -	Lmatloc[1 * width + gx -1 + k]) +
						abs(Rmatloc[1 * width + gx] -		Lmatloc[1 * width + gx + k]) + 
						abs(Rmatloc[1 * width + gx + 1] - Lmatloc[1 * width + gx + 1 + k]) + 
						abs(Rmatloc[1 * width + gx + 2] - Lmatloc[1 * width + gx + 2 + k]);
				winsum += abs(Rmatloc[2 * width + gx -2] -		Lmatloc[2 * width + gx -2 + k]) + 
						abs(Rmatloc[2 * width + gx -1] -		Lmatloc[2 * width + gx -1 + k]) +
						abs(Rmatloc[2 * width + gx] -			Lmatloc[2 * width + gx + k]) + 
						abs(Rmatloc[2 * width + gx + 1] -		Lmatloc[2 * width + gx + 1 + k]) + 
						abs(Rmatloc[2 * width + gx + 2] -		Lmatloc[2 * width + gx + 2 + k]);
				winsum += abs(Rmatloc[3 * width + gx -2] -	Lmatloc[3 * width + gx -2 + k]) + 
						abs(Rmatloc[3 * width + gx -1] -	Lmatloc[3 * width + gx -1 + k]) +
						abs(Rmatloc[3 * width + gx] -		Lmatloc[3 * width + gx + k]) + 
						abs(Rmatloc[3 * width + gx + 1] - Lmatloc[3 * width + gx + 1 + k]) + 
						abs(Rmatloc[3 * width + gx + 2] - Lmatloc[3 * width + gx + 2 + k]);
				winsum += abs(Rmatloc[4 * width + gx -2] -	Lmatloc[4 * width + gx -2 + k]) + 
						abs(Rmatloc[4 * width + gx -1] -	Lmatloc[4 * width + gx -1 + k]) +
						abs(Rmatloc[4 * width + gx] -		Lmatloc[4 * width + gx + k]) + 
						abs(Rmatloc[4 * width + gx + 1] - Lmatloc[4 * width + gx + 1 + k]) + 
						abs(Rmatloc[4 * width + gx + 2] - Lmatloc[4 * width + gx + 2 + k]);
			} 
			// unrolled loop for global implementation
			else {
				width = paddeddim[1];
				winsum = abs(Rmat[(gy - 2) * width + gx -2] -	Lmat[(gy - 2) * width + gx -2 + k]) + 
					abs(Rmat[(gy - 2) * width + gx -1] -	Lmat[(gy - 2) * width + gx -1 + k]) +
					abs(Rmat[(gy - 2) * width + gx] -		Lmat[(gy - 2) * width + gx + k]) + 
					abs(Rmat[(gy - 2) * width + gx + 1] - Lmat[(gy - 2) * width + gx + 1 + k]) + 
					abs(Rmat[(gy - 2) * width + gx + 2] - Lmat[(gy - 2) * width + gx + 2 + k]);
				winsum += abs(Rmat[(gy - 1) * width + gx -2] -	Lmat[(gy - 1) * width + gx -2 + k]) + 
						abs(Rmat[(gy - 1) * width + gx -1] -	Lmat[(gy - 1) * width + gx -1 + k]) +
						abs(Rmat[(gy - 1) * width + gx] -		Lmat[(gy - 1) * width + gx + k]) + 
						abs(Rmat[(gy - 1) * width + gx + 1] - Lmat[(gy - 1) * width + gx + 1 + k]) + 
						abs(Rmat[(gy - 1) * width + gx + 2] - Lmat[(gy - 1) * width + gx + 2 + k]);
				winsum += abs(Rmat[gy * width + gx -2] -		Lmat[gy * width + gx -2 + k]) + 
						abs(Rmat[gy * width + gx -1] -		Lmat[gy * width + gx -1 + k]) +
						abs(Rmat[gy * width + gx] -			Lmat[gy * width + gx + k]) + 
						abs(Rmat[gy * width + gx + 1] -		Lmat[gy * width + gx + 1 + k]) + 
						abs(Rmat[gy * width + gx + 2] -		Lmat[gy * width + gx + 2 + k]);
				winsum += abs(Rmat[(gy + 1) * width + gx -2] -	Lmat[(gy + 1) * width + gx -2 + k]) + 
						abs(Rmat[(gy + 1) * width + gx -1] -	Lmat[(gy + 1) * width + gx -1 + k]) +
						abs(Rmat[(gy + 1) * width + gx] -		Lmat[(gy + 1) * width + gx + k]) + 
						abs(Rmat[(gy + 1) * width + gx + 1] - Lmat[(gy + 1) * width + gx + 1 + k]) + 
						abs(Rmat[(gy + 1) * width + gx + 2] - Lmat[(gy + 1) * width + gx + 2 + k]);
				winsum += abs(Rmat[(gy + 2) * width + gx -2] -	Lmat[(gy + 2) * width + gx -2 + k]) + 
						abs(Rmat[(gy + 2) * width + gx -1] -	Lmat[(gy + 2) * width + gx -1 + k]) +
						abs(Rmat[(gy + 2) * width + gx] -		Lmat[(gy + 2) * width + gx + k]) + 
						abs(Rmat[(gy + 2) * width + gx + 1] - Lmat[(gy + 2) * width + gx + 1 + k]) + 
						abs(Rmat[(gy + 2) * width + gx + 2] - Lmat[(gy + 2) * width + gx + 2 + k]);
			}
			//if(gx == 2 && gy == 8)
				//printf("k: %d, winsum: %d\n", k, winsum);
			wink = winsum < temp ? k : wink;
			temp = winsum < temp ? winsum : temp;
		}

		SADmat[gy * paddeddim[1] + gx] = wink;

		//printf("reached the end: gx: %d gy: %d\n", gx, gy);
		//if(gx == (imdim[1] - dispLevel[0] - rad[0]))
			last[gy * numGroupsX + ga] = true;
		//printf("last[(gy - 0) * numGroupsX + ga]: %d\n", last[(gy - 0) * numGroupsX + ga]);
	}

}