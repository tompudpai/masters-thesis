
// TODO: Add OpenCL kernel code here.

__kernel void copyToLocal()
{
	printf("useless kernel\n"); 
}

__kernel void sad(
	__global uchar* Rmat,
	__global uchar* Lmat,
	__global int* SADmat,
	__global int* imdim,
	__global int* paddeddim,
	__global bool* first,
	__local uchar *Rmatloc,
	__local uchar *Lmatloc,
	int local_size,
	__global bool* local_bool,
	__global int* localItemSize,
	__global bool* unroll,
	__global int* dispLevel,
	__global int* rad,
	__global bool* rowDone
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
	int wx = get_group_id(0);
	int wy = get_group_id(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	//if (gx < 5 && gy < 5){ 
		//printf("global(%d, %d)| group(%d, %d) | local(%d, %d)\n", gx, gy, wx, wy, lx, ly);
	//}

	/*for (int i=0; i < local_size; i++) {
		if( gx < 5 && gy < 5 && i < 3)
			printf("global(%d, %d)| group(%d, %d) | local(%d, %d) | Rmat[%d]: %d\n", 
				gx, gy, wx, wy, lx, ly, i, Rmat[i]);
		//if( gx == 0 && gy == 0)
			//p[i] = i;
	}*/

	__private int LOCAL_MEM_HEIGHT = 5;
	__private int LOCAL_MEM_WIDTH = 512;

	if(gx < rad[0] || gx > imdim[1] - dispLevel[0] - rad[0] || // 2 <= gx <= 368
		gy < rad[0] || gy >= imdim[0] - rad[0]) return; // 2 <= gy <= 380
	else{

		__private int i, j, m;
		__private int leftBound, rightBound, topBound;
		__private int numGroupsX = paddeddim[1]/localItemSize[1];
		
		//if(gx == 307 && gy == 92)
			//printf("Point 1\n");		
		// load local memory
		if(local_bool[0]){
		//if(gx == 307 && gy == 92)
			//printf("Point 2\n");
			// load local memory in parallel
			if(local_bool[1]) {
				// determine boundaries of work-item IDs in workgroup
				if (wx == 0) // first workgroup
					leftBound = rad[0];
				else
					leftBound = wx * localItemSize[1];
				
				if (wx == (int)((imdim[1] - dispLevel[0] - rad[0])/localItemSize[1])) // last workgroup
					rightBound = imdim[1] - dispLevel[0] - rad[0];
				else
					rightBound = (wx + 1) * localItemSize[1] - 1;

				//if ((localItemSize[1] < rad[0] && wy < rad[0]) || wy == 0)
				if (wy == 0)
					topBound = rad[0];
				else
					topBound = wy * localItemSize[0];

				// populate local memory

				if(gy == topBound){ 
					__private int a;
					
					for(a = 0; a < 5; a++){ 
						Rmatloc[a * LOCAL_MEM_WIDTH + gx] = Rmat[(gy - 2 + a) * paddeddim[1] + gx];
						Lmatloc[a * LOCAL_MEM_WIDTH + gx] = Lmat[(gy - 2 + a) * paddeddim[1] + gx];
						Lmatloc[a * LOCAL_MEM_WIDTH + gx + 65] = Lmat[(gy - 2 + a) * paddeddim[1] + gx + 65];
						if(gx == leftBound){
							Rmatloc[a * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy - 2 + a) * paddeddim[1] + gx - 1];
							Rmatloc[a * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy - 2 + a) * paddeddim[1] + gx - 2];
							Lmatloc[a * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy - 2 + a) * paddeddim[1] + gx - 1];
							Lmatloc[a * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy - 2 + a) * paddeddim[1] + gx - 2];
						}
						if(gx == rightBound){ 
							Rmatloc[a * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy - 2 + a) * paddeddim[1] + gx + 1];
							Rmatloc[a * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy - 2 + a) * paddeddim[1] + gx + 2];
						}
					}
					/*
					// left and right matrices within workgroup boundaries
					Rmatloc[0 * LOCAL_MEM_WIDTH + gx] = Rmat[(gy - 2) * paddeddim[1] + gx];
					Rmatloc[1 * LOCAL_MEM_WIDTH + gx] = Rmat[(gy - 1) * paddeddim[1] + gx];
					Rmatloc[2 * LOCAL_MEM_WIDTH + gx] = Rmat[(gy - 0) * paddeddim[1] + gx];
					Rmatloc[3 * LOCAL_MEM_WIDTH + gx] = Rmat[(gy + 1) * paddeddim[1] + gx];
					Rmatloc[4 * LOCAL_MEM_WIDTH + gx] = Rmat[(gy + 2) * paddeddim[1] + gx];	
					Lmatloc[0 * LOCAL_MEM_WIDTH + gx] = Lmat[(gy - 2) * paddeddim[1] + gx];
					Lmatloc[1 * LOCAL_MEM_WIDTH + gx] = Lmat[(gy - 1) * paddeddim[1] + gx];
					Lmatloc[2 * LOCAL_MEM_WIDTH + gx] = Lmat[(gy - 0) * paddeddim[1] + gx];
					Lmatloc[3 * LOCAL_MEM_WIDTH + gx] = Lmat[(gy + 1) * paddeddim[1] + gx];
					Lmatloc[4 * LOCAL_MEM_WIDTH + gx] = Lmat[(gy + 2) * paddeddim[1] + gx];
					//if(gx > rightBound - 65){ 
						Lmatloc[0 * LOCAL_MEM_WIDTH + gx + 65] = Lmat[(gy - 2) * paddeddim[1] + gx + 65];
						Lmatloc[1 * LOCAL_MEM_WIDTH + gx + 65] = Lmat[(gy - 1) * paddeddim[1] + gx + 65];
						Lmatloc[2 * LOCAL_MEM_WIDTH + gx + 65] = Lmat[(gy - 0) * paddeddim[1] + gx + 65];
						Lmatloc[3 * LOCAL_MEM_WIDTH + gx + 65] = Lmat[(gy + 1) * paddeddim[1] + gx + 65];
						Lmatloc[4 * LOCAL_MEM_WIDTH + gx + 65] = Lmat[(gy + 2) * paddeddim[1] + gx + 65];
					//}

					// boundary
					if(gx == leftBound){ 
						Rmatloc[0 * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy - 2) * paddeddim[1] + gx - 1];
						Rmatloc[1 * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy - 1) * paddeddim[1] + gx - 1];
						Rmatloc[2 * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy - 0) * paddeddim[1] + gx - 1];
						Rmatloc[3 * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy + 1) * paddeddim[1] + gx - 1];
						Rmatloc[4 * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy + 2) * paddeddim[1] + gx - 1];
						Rmatloc[0 * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy - 2) * paddeddim[1] + gx - 2];
						Rmatloc[1 * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy - 1) * paddeddim[1] + gx - 2];
						Rmatloc[2 * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy - 0) * paddeddim[1] + gx - 2];
						Rmatloc[3 * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy + 1) * paddeddim[1] + gx - 2];
						Rmatloc[4 * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy + 2) * paddeddim[1] + gx - 2];
						Lmatloc[0 * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy - 2) * paddeddim[1] + gx - 1];
						Lmatloc[1 * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy - 1) * paddeddim[1] + gx - 1];
						Lmatloc[2 * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy - 0) * paddeddim[1] + gx - 1];
						Lmatloc[3 * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy + 1) * paddeddim[1] + gx - 1];
						Lmatloc[4 * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy + 2) * paddeddim[1] + gx - 1];
						Lmatloc[0 * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy - 2) * paddeddim[1] + gx - 2];
						Lmatloc[1 * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy - 1) * paddeddim[1] + gx - 2];
						Lmatloc[2 * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy - 0) * paddeddim[1] + gx - 2];
						Lmatloc[3 * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy + 1) * paddeddim[1] + gx - 2];
						Lmatloc[4 * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy + 2) * paddeddim[1] + gx - 2];
					}
					if(gx == rightBound) { 
						Rmatloc[0 * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy - 2) * paddeddim[1] + gx + 1];
						Rmatloc[1 * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy - 1) * paddeddim[1] + gx + 1];
						Rmatloc[2 * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy - 0) * paddeddim[1] + gx + 1];
						Rmatloc[3 * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy + 1) * paddeddim[1] + gx + 1];
						Rmatloc[4 * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy + 2) * paddeddim[1] + gx + 1];
						Rmatloc[0 * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy - 2) * paddeddim[1] + gx + 2];
						Rmatloc[1 * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy - 1) * paddeddim[1] + gx + 2];
						Rmatloc[2 * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy - 0) * paddeddim[1] + gx + 2];
						Rmatloc[3 * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy + 1) * paddeddim[1] + gx + 2];
						Rmatloc[4 * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy + 2) * paddeddim[1] + gx + 2];
						//for(i = rightBound + 1; i < rightBound + dispLevel[0] + rad[0]; i++) {
							//Lmatloc[0 * LOCAL_MEM_WIDTH + i] = Lmat[(gy - 2) * paddeddim[1] + i];
							//Lmatloc[1 * LOCAL_MEM_WIDTH + i] = Lmat[(gy - 1) * paddeddim[1] + i];
							//Lmatloc[2 * LOCAL_MEM_WIDTH + i] = Lmat[(gy - 0) * paddeddim[1] + i];
							//Lmatloc[3 * LOCAL_MEM_WIDTH + i] = Lmat[(gy + 1) * paddeddim[1] + i];
							//Lmatloc[4 * LOCAL_MEM_WIDTH + i] = Lmat[(gy + 2) * paddeddim[1] + i];
						//}
					}*/
				}
				else { 
					while(!rowDone[(gy - 1) * numGroupsX + wx]) {
						//printf("waiting on rowDone[%d, %d]: %d\n", gy - 1, wx, rowDone[(gy - 1) * numGroupsX + wx]);
						printf("");
					}
					Rmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx] = Rmat[(gy + rad[0]) * paddeddim[1] + gx];
					Lmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx] = Lmat[(gy + rad[0]) * paddeddim[1] + gx];
					if(gx == leftBound){ 
						Rmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx - 2] = Rmat[(gy + rad[0]) * paddeddim[1] + gx - 2];
						Rmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx - 1] = Rmat[(gy + rad[0]) * paddeddim[1] + gx - 1];
						Lmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx - 2] = Lmat[(gy + rad[0]) * paddeddim[1] + gx - 2];		
						Lmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx - 1] = Lmat[(gy + rad[0]) * paddeddim[1] + gx - 1];		
					}	
					if(gx == rightBound){ 
						Rmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx + 1] = Rmat[(gy + rad[0]) * paddeddim[1] + gx + 1];
						Rmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + gx + 2] = Rmat[(gy + rad[0]) * paddeddim[1] + gx + 2];
						for(i = rightBound + 1; i < rightBound + dispLevel[0] + rad[0]; i++) 
							Lmatloc[((gy - topBound - 1) % LOCAL_MEM_HEIGHT) * LOCAL_MEM_WIDTH + i] = Lmat[(gy + rad[0]) * paddeddim[1] + i];
					}			
				}
			}
			// load local memory (by first work-item)
			else { // check if local mem of last row copied yet
				//if(gx == 307 && gy == 92)
					//printf("Point 3\n");
				// determine boundaries of local memory copied
				if (wx == 0) // first workgroup in row
					leftBound = 0;
				else
					leftBound = wx * localItemSize[1] - rad[0];

				if (wx == (int)((imdim[1] - dispLevel[0] - rad[0])/localItemSize[1])) // last workgroup in row
					rightBound = imdim[1] - dispLevel[0] + 1; //371
					//rightBound = imdim[1] - dispLevel[0] - rad[0];
				else
					rightBound = (wx + 1) * localItemSize[1] + rad[0] - 1;

				// top boundary is work-item ID
				/*if(localItemSize[1] == 1 && wy == 2)
					topBound = rad[0];
				else if (localItemSize[1] == 2 && wy == 1)
					topBound = rad[0];
				else if (wy == 0)
					topBound = rad[0];*/
				//if ((localItemSize[0] < rad[0] && wy < rad[0]) || wy == 0)
				if(wy == 0)
					topBound = rad[0];
				else
					topBound = wy * localItemSize[0];
	
				//printf("topBound for (%d,%d): %d\n",gx,gy,topBound);
				// populate local memory 

				if(gy == topBound){ 
					for (i = 0; i < LOCAL_MEM_HEIGHT; i++) {
						for (j = leftBound; j <= rightBound; j++){ 
							Rmatloc[i * LOCAL_MEM_WIDTH + j] = Rmat[(gy - rad[0] + i) * paddeddim[1] + j];
						}
						for (m = leftBound; m <= rightBound + dispLevel[0] - 1; m++){ 
							Lmatloc[i * LOCAL_MEM_WIDTH + m] = Lmat[(gy - rad[0] + i) * paddeddim[1] + m];
						}
					}
				}
				else{ 
					//rowDone[(gy - 1) * numGroupsX + wx] = true;
					//printf("rowDone[%d, %d]: %d\n", gy - 1, wx, rowDone[(gy - 1) * numGroupsX + wx]);
					while(!rowDone[(gy - 1) * numGroupsX + wx]) {
						printf("");
					}
					__private int localMemRow = (gy - topBound - 1) % LOCAL_MEM_HEIGHT;
					for (j = leftBound; j <= rightBound; j++){ 
						Rmatloc[localMemRow * LOCAL_MEM_WIDTH + j] = Rmat[(gy + rad[0]) * paddeddim[1] + j];
						//if(gx == 307 && gy == 92)
							//printf("Rmatloc[%d]: %d, ",localMemRow * LOCAL_MEM_WIDTH + j,Rmatloc[localMemRow * LOCAL_MEM_WIDTH + j]);
						}
					for (m = leftBound; m <= rightBound + dispLevel[0] - 1; m++){ 
						Lmatloc[localMemRow * LOCAL_MEM_WIDTH + m] = Lmat[(gy + rad[0]) * paddeddim[1] + m];
						//if(gx == 307 && gy == 92)	
							//printf("Lmatloc[%d]: %d, ",localMemRow * LOCAL_MEM_WIDTH + m,Lmatloc[localMemRow * LOCAL_MEM_WIDTH + m]);
						}
					}
					
				//first[gy * numGroupsX + wx] = true;
			}
			//printf("local memory loaded: global(%d) | group(%d) | local(%d)\n", gx, wx, lx);
		}


		// synchronize workgroup when loading in parallel
		//if(local_bool[0] && local_bool[1])
			barrier(CLK_LOCAL_MEM_FENCE);

		__private int k, h, w, winsum;
		__private int temp, width, wink, winlength;
	
		width = LOCAL_MEM_WIDTH;//paddeddim[1];

		winlength = 2 * rad[0] + 1;
		temp = 255 * winlength * winlength;
		wink = 0;

		for(k = 0; k < dispLevel[0]; k++){
			winsum = 0;

			// loop for SAD calculation (local and global implementations)
			if(!unroll[0]){
				if(local_bool[0])
					for(h = 0; h < LOCAL_MEM_HEIGHT; h++){ 
						for(w = gx - rad[0]; w < gx + rad[0] + 1; w++){
							winsum += abs(Rmatloc[h * LOCAL_MEM_WIDTH + w] - Lmatloc[h * LOCAL_MEM_WIDTH + w + k]);
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
				
				winsum = abs(Rmatloc[0 * width + gx -2] - Lmatloc[0 * width + gx -2 + k]) + 
						abs(Rmatloc[0 * width + gx -1] -  Lmatloc[0 * width + gx -1 + k]) +
						abs(Rmatloc[0 * width + gx] -	  Lmatloc[0 * width + gx + k]) + 
						abs(Rmatloc[0 * width + gx + 1] - Lmatloc[0 * width + gx + 1 + k]) + 
						abs(Rmatloc[0 * width + gx + 2] - Lmatloc[0 * width + gx + 2 + k]) +
						abs(Rmatloc[1 * width + gx -2] -  Lmatloc[1 * width + gx -2 + k]) + 
						abs(Rmatloc[1 * width + gx -1] -  Lmatloc[1 * width + gx -1 + k]) +
						abs(Rmatloc[1 * width + gx] -     Lmatloc[1 * width + gx + k]) + 
						abs(Rmatloc[1 * width + gx + 1] - Lmatloc[1 * width + gx + 1 + k]) + 
						abs(Rmatloc[1 * width + gx + 2] - Lmatloc[1 * width + gx + 2 + k]) +
						abs(Rmatloc[2 * width + gx -2] -  Lmatloc[2 * width + gx -2 + k]) + 
						abs(Rmatloc[2 * width + gx -1] -  Lmatloc[2 * width + gx -1 + k]) +
						abs(Rmatloc[2 * width + gx] - 	  Lmatloc[2 * width + gx + k]) + 
						abs(Rmatloc[2 * width + gx + 1] - Lmatloc[2 * width + gx + 1 + k]) + 
						abs(Rmatloc[2 * width + gx + 2] - Lmatloc[2 * width + gx + 2 + k]) +
						abs(Rmatloc[3 * width + gx -2] -  Lmatloc[3 * width + gx -2 + k]) + 
						abs(Rmatloc[3 * width + gx -1] -  Lmatloc[3 * width + gx -1 + k]) +
						abs(Rmatloc[3 * width + gx] -	  Lmatloc[3 * width + gx + k]) + 
						abs(Rmatloc[3 * width + gx + 1] - Lmatloc[3 * width + gx + 1 + k]) + 
						abs(Rmatloc[3 * width + gx + 2] - Lmatloc[3 * width + gx + 2 + k]) +
						abs(Rmatloc[4 * width + gx -2] -  Lmatloc[4 * width + gx -2 + k]) + 
						abs(Rmatloc[4 * width + gx -1] -  Lmatloc[4 * width + gx -1 + k]) +
						abs(Rmatloc[4 * width + gx] -	  Lmatloc[4 * width + gx + k]) + 
						abs(Rmatloc[4 * width + gx + 1] - Lmatloc[4 * width + gx + 1 + k]) + 
						abs(Rmatloc[4 * width + gx + 2] - Lmatloc[4 * width + gx + 2 + k]);
			} 
			// unrolled loop for global implementation
			else {
				width = paddeddim[1];
				winsum = abs(Rmat[(gy - 2) * width + gx -2] - Lmat[(gy - 2) * width + gx -2 + k]) + 
						abs(Rmat[(gy - 2) * width + gx -1] -  Lmat[(gy - 2) * width + gx -1 + k]) +
						abs(Rmat[(gy - 2) * width + gx] -	  Lmat[(gy - 2) * width + gx + k]) + 
						abs(Rmat[(gy - 2) * width + gx + 1] - Lmat[(gy - 2) * width + gx + 1 + k]) + 
						abs(Rmat[(gy - 2) * width + gx + 2] - Lmat[(gy - 2) * width + gx + 2 + k]) +
						abs(Rmat[(gy - 1) * width + gx -2] -  Lmat[(gy - 1) * width + gx -2 + k]) + 
						abs(Rmat[(gy - 1) * width + gx -1] -  Lmat[(gy - 1) * width + gx -1 + k]) +
						abs(Rmat[(gy - 1) * width + gx] -     Lmat[(gy - 1) * width + gx + k]) + 
						abs(Rmat[(gy - 1) * width + gx + 1] - Lmat[(gy - 1) * width + gx + 1 + k]) + 
						abs(Rmat[(gy - 1) * width + gx + 2] - Lmat[(gy - 1) * width + gx + 2 + k]) +
						abs(Rmat[gy * width + gx -2] -		  Lmat[gy * width + gx -2 + k]) + 
						abs(Rmat[gy * width + gx -1] -		  Lmat[gy * width + gx -1 + k]) +
						abs(Rmat[gy * width + gx] -			  Lmat[gy * width + gx + k]) + 
						abs(Rmat[gy * width + gx + 1] -		  Lmat[gy * width + gx + 1 + k]) + 
						abs(Rmat[gy * width + gx + 2] -		  Lmat[gy * width + gx + 2 + k]) +
						abs(Rmat[(gy + 1) * width + gx -2] -  Lmat[(gy + 1) * width + gx -2 + k]) + 
						abs(Rmat[(gy + 1) * width + gx -1] -  Lmat[(gy + 1) * width + gx -1 + k]) +
						abs(Rmat[(gy + 1) * width + gx] -	  Lmat[(gy + 1) * width + gx + k]) + 
						abs(Rmat[(gy + 1) * width + gx + 1] - Lmat[(gy + 1) * width + gx + 1 + k]) + 
						abs(Rmat[(gy + 1) * width + gx + 2] - Lmat[(gy + 1) * width + gx + 2 + k]) +
						abs(Rmat[(gy + 2) * width + gx -2] -  Lmat[(gy + 2) * width + gx -2 + k]) + 
						abs(Rmat[(gy + 2) * width + gx -1] -  Lmat[(gy + 2) * width + gx -1 + k]) +
						abs(Rmat[(gy + 2) * width + gx] -	  Lmat[(gy + 2) * width + gx + k]) + 
						abs(Rmat[(gy + 2) * width + gx + 1] - Lmat[(gy + 2) * width + gx + 1 + k]) + 
						abs(Rmat[(gy + 2) * width + gx + 2] - Lmat[(gy + 2) * width + gx + 2 + k]);
			}
			//if(gx == 307 && gy == 92)
				//printf("k: %d, winsum: %d\n", k, winsum);
			wink = winsum < temp ? k : wink;
			temp = winsum < temp ? winsum : temp;
		}

		SADmat[gy * paddeddim[1] + gx] = wink;

		//printf("reached the end: gx: %d gy: %d\n", gx, gy);
		//if(gx == (imdim[1] - dispLevel[0] - rad[0]))

		__private int lastInRow;
		if (wx == (int)((imdim[1] - dispLevel[0] - rad[0])/localItemSize[1])) // last workgroup in row
			lastInRow = imdim[1] - dispLevel[0] - rad[0];
		else
			lastInRow = (wx + 1) * localItemSize[1] - 1;
		if(gx == lastInRow)
			rowDone[gy * numGroupsX + wx] = true;
	    //printf("rowDone, lastInRow for (%d,%d): %d\n",gx,gy,rowDone[gy * numGroupsX + wx],lastInRow);
		//if(gx == 307 && gy == 92)
			//printf("gx: %d, lastInRow: %d\n", gx, lastInRow);
		//printf("rowDone[(gy - 0) * numGroupsX + wx]: %d\n", rowDone[(gy - 0) * numGroupsX + wx]);
		//if(gx == 307 && gy == 92)
			//printf("global(%d, %d)| group(%d, %d) | local(%d, %d)\n", gx, gy, wx, wy, lx, ly);
	}

}