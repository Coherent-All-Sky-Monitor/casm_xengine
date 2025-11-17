/*gcc -o beamformer beamformer.c -I/usr/local/include -L/usr/local/lib -lm -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran
python beamformer was too slow, decided to use python to write up header etc but do actual beamforming in C.
This code should take 8 parameters:
* data file name
* calibration file name
* number of antennas in voltage file
* number of antennas to use in beamforming
* start frequency
* separation
* beam number
* output file name
assumes 48 channels for beamformer (weights for 8 data channels), ONLY 1 beam
greg hellbourg
ghellbourg@astro.caltech.edu
*/


#include "stdio.h"
#include "stdlib.h"
#include "sys/types.h"
#include "sys/socket.h"
#include "string.h"
#include "netinet/in.h"
#include "netdb.h"
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <math.h>

int NW = 48;    // number of channels for the beamformer
float PI = 3.141592653589793238;
float CVAC = 299792458.0;
int NANT = 96;


int init_weights(char * fnam, char *flagants, float *antpos, float *weights, int nPols) {

        // assumes NANT antennas
        // antpos: takes only easting
        // weights: takes [ant, NW==48]

        FILE *fin;
        FILE *f_used_ants;
        int rd;

        // --- MODIFICATION START ---
        // The file now contains a list of antennas TO USE.
        int used_ants[NANT], n_used=0;
        f_used_ants = fopen(flagants,"r");
        if (f_used_ants == NULL) {
            perror("Error opening antenna list file");
            return 1; // Return an error
        }
        // Read all antenna numbers from the file
        while (!feof(f_used_ants) && n_used < NANT) {
            if (fscanf(f_used_ants, "%d\n", &used_ants[n_used]) == 1) {
                n_used++;
            }
        }
        fclose(f_used_ants);
        // --- MODIFICATION END ---
	
        fin=fopen(fnam,"rb");
        if (fin == NULL) {
            perror("Error opening calibration weight file");
            return 1; // Return an error
        }
        rd = fread(antpos,2*NANT*sizeof(float),1,fin);
        rd = fread(weights,NANT*NW*nPols*2*sizeof(float),1,fin);
        fclose(fin);

        float wnorm;
        int i;

        // --- MODIFICATION START ---
        // Loop over all possible antennas (0 to NANT-1)
        for (int ii=0; ii<NANT; ii++) {
            
            // Check if the current antenna 'ii' is in our list of antennas to use
            int is_used = 0; // Flag to check if antenna should be used
            for (int kk=0; kk<n_used; kk++) {
                if (used_ants[kk] == ii) {
                    is_used = 1;
                    break; // Found it, no need to search further
                }
            }

            // Now, apply the weights based on whether the antenna is used or not
            for (int jj=0; jj<NW*nPols; jj++) {
                i = ii*NW*nPols+jj;

                if (is_used) {
                    // This antenna IS in the list, so we use its calibration weights
                    // and normalize them to unit length.
                    wnorm = sqrt(weights[2*i]*weights[2*i] + weights[2*i+1]*weights[2*i+1]);
                    if (wnorm != 0.0) {
                        weights[2*i] /= wnorm;
                        weights[2*i+1] /= wnorm;
                    }
                } else {
                    // This antenna IS NOT in the list, so we zero out its weights.
                    weights[2*i] = 0.0;
                    weights[2*i+1] = 0.0;
                }
            }
        }
        // --- MODIFICATION END ---

        return 0;
}

int init_weights_old(char * fnam, char *flagants, float *antpos, float *weights, int nPols) {

        // assumes NANT antennas
        // antpos: takes only easting
        // weights: takes [ant, NW==48]

        FILE *fin;
	FILE *fflag;
        FILE *fants;
        int rd;

	int flags[NANT], nflag=0;
        fflag = fopen(flagants,"r");
	while (!feof(fflag)) {
	  fscanf(fflag,"%d\n",&flags[nflag]);
	  nflag++;
	}
	fclose(fflag);	 
	
        fin=fopen(fnam,"rb");

        rd = fread(antpos,2*NANT*sizeof(float),1,fin);
        rd = fread(weights,NANT*NW*nPols*2*sizeof(float),1,fin);
        float wnorm;
	int i;
	for (int ii=0;ii<NANT;ii++) {
	  for (int jj=0;jj<NW*nPols;jj++) {
	    i = ii*NW*nPols+jj;
	    wnorm = sqrt(weights[2*i]*weights[2*i] + weights[2*i+1]*weights[2*i+1]);
	    for (int kk=0;kk<nflag;kk++) {
	      if (flags[kk]==ii) {
		weights[2*i] = 0.;
		weights[2*i+1] = 0.;
	      }
	    }
	    if (wnorm!=0.0) {
	      weights[2*i] /= wnorm;
	      weights[2*i+1] /= wnorm;
	    }
	  }
	}

        fclose(fin);
        return 0;

}

void calc_weights(float *antpos, float *weights, float *freqs, float *wr, float *wi, float sep, int nBeamNum, int nPols) {


        float theta, afac, twr, twi;

        theta = sep*(127.-(float)nBeamNum)*PI/10800.; // radians
        for(int nAnt=0;nAnt<NANT;nAnt++){
                for(int nChan=0;nChan<48;nChan++){
                        for(int nPol=0;nPol<nPols;nPol++){
                                afac = -2.*PI*freqs[nChan*8+4]*theta/CVAC; // factor for rotate
                                twr = cos(afac*antpos[nAnt]);
                                twi = sin(afac*antpos[nAnt]);

                                wr[nAnt*(48*nPols)+nChan*nPols+nPol] = (twr*weights[(nAnt*(48*nPols)+nChan*nPols+nPol)*2] - twi*weights[(nAnt*(48*nPols)+nChan*nPols+nPol)*2+1]);
                                wi[nAnt*(48*nPols)+nChan*nPols+nPol] = (twi*weights[(nAnt*(48*nPols)+nChan*nPols+nPol)*2] + twr*weights[(nAnt*(48*nPols)+nChan*nPols+nPol)*2+1]);
                        }
                }
        }

/* write beamformer weights, if needed -- can be made optional? */
/*
        FILE *write_ptr;
        char filename[100];
        sprintf(filename, "/home/user/T3_detect/testdir/beamweights_%d", nBeamNum);
        write_ptr = fopen(filename,"wb");
        fwrite(wr,sizeof(float),64*48*2,write_ptr);
        fwrite(wi,sizeof(float),64*48*2,write_ptr);
        fclose(write_ptr);
        printf("wrote beamformer weights -- size = %d\n",(int)sizeof(float));
        printf("data written as int -- size = %d\n",(int)sizeof(int));
*/

}

void beamformer(char *input, float *wr, float *wi, unsigned char *output, int nChans, int nTimes, int nPols, int incoh) {

        float inr_x, ini_x, inr_y, ini_y;
        float wrx, wix, wry, wiy;
        float rx, ix, ry, iy;
        float tmprealX, tmpimagX, tmprealY, tmpimagY;
	char v;

        for(int nTime=0;nTime<nTimes;nTime++){
                for(int nChan=0;nChan<48;nChan++){
                        for(int i=0;i<8;i++){
                                rx = 0;
                                ix = 0;
                                ry = 0;
                                iy = 0;
                                for(int nAnt=0;nAnt<NANT/2;nAnt++){
				  v = input[nAnt*(nChans*nPols*nTimes)+(nChan*8+i)*(nPols*nTimes)+nTime*2];				  
				  inr_x = (float)((char)(((unsigned char)(v) & (unsigned char)(15)) << 4) >> 4);
				  //inr_x = (float)(((char)((v & 15) << 4)) >> 4);
				  v = input[nAnt*(nChans*nPols*nTimes)+(nChan*8+i)*(nPols*nTimes)+nTime*2];
				  ini_x = (float)((char)(((unsigned char)(v) & (unsigned char)(240))) >> 4);
				  //ini_x = (float)(((char)((v & 240))) >> 4);
				  v = input[nAnt*(nChans*nPols*nTimes)+(nChan*8+i)*(nPols*nTimes)+nTime*2+1];
				  inr_y = (float)((char)(((unsigned char)(v) & (unsigned char)(15)) << 4) >> 4);
				  //inr_y = (float)(((char)((v & 15) << 4)) >> 4);
				  v = input[nAnt*(nChans*nPols*nTimes)+(nChan*8+i)*(nPols*nTimes)+nTime*2+1];
				  ini_y = (float)((char)(((unsigned char)(v) & (unsigned char)(240))) >> 4);
				  //ini_y = (float)(((char)((v & 240))) >> 4);

                                        /***********/
                                        /*towrite[nAnt*(nChans*nTimes*nPols*2)+(nChan*8+i)*(nPols*nTimes*2)+nTime*nPols*2  ] = (int)inr_x;
                                        towrite[nAnt*(nChans*nTimes*nPols*2)+(nChan*8+i)*(nPols*nTimes*2)+nTime*nPols*2+1] = (int)ini_x;
                                        towrite[nAnt*(nChans*nTimes*nPols*2)+(nChan*8+i)*(nPols*nTimes*2)+nTime*nPols*2+2] = (int)inr_y;
                                        towrite[nAnt*(nChans*nTimes*nPols*2)+(nChan*8+i)*(nPols*nTimes*2)+nTime*nPols*2+3] = (int)ini_y;*/
                                        /***********/

                                        wrx = wr[nAnt*(48*nPols)+nChan*nPols];
                                        wix = wi[nAnt*(48*nPols)+nChan*nPols];
                                        wry = wr[nAnt*(48*nPols)+nChan*nPols+1];
                                        wiy = wi[nAnt*(48*nPols)+nChan*nPols+1];

					if (!incoh) {
					  rx += inr_x*wrx - ini_x*wix;
					  ix += inr_x*wix + ini_x*wrx;
					  ry += inr_y*wry - ini_y*wiy;
					  iy += inr_y*wiy + ini_y*wry;
					}
					else {
					  rx += inr_x*inr_x*wrx*wrx + inr_y*inr_y*wry*wry + ini_x*ini_x*wix*wix + ini_y*ini_y*wiy*wiy;
					}

                                }
				if (!incoh)
				  output[nTime*nChans+nChan*8+i] = (unsigned char)(128.*16.*(rx*rx + ix*ix + ry*ry + iy*iy) / NANT / NANT);
				else
				  output[nTime*nChans+nChan*8+i] = (unsigned char)(64.*64.*rx/NANT/NANT);
                        }
                }
        }
}

void usage()
{
  fprintf (stdout,
           "t3_beamformer [options]\n"
           " -d voltage data file name [no default]\n"
           " -f calibration file name [no default]\n"
           " -o output file name [no default]\n"
		   " -a number of antennas in file [default 30]\n"
		   " -u number of antennas to be used [default 24]\n"
           " -z fch1 in MHz [default 1530]\n"
           " -s interbeam separation in arcmin [default 1.4]\n"
           " -n beam number [0 -- 255, default 127]\n"
	   " -i incoherent beamforming\n"
	   " -q flagants file [no default]\n"
           " -h print usage\n");
}


int main (int argc, char *argv[]) {


        int nChans = 384;
        int nPols = 2;
        int nTimes = 2;
	int incoh = 0;

        // read params : fch1, fnam, fdataname, sep
        int arg = 0;
        float fch1 = 1530.0;
        float sep = 1.0;
        int nBeamNum = 127, nUsedAnts;
        char * fnam;
        fnam=(char *)malloc(sizeof(char)*200);
        char * fflag;
        fflag=(char *)malloc(sizeof(char)*200);
        sprintf(fnam,"nofile");
        char * fdata;
        fdata=(char *)malloc(sizeof(char)*200);
        sprintf(fdata,"nofile");
        char * fout;
        fout=(char *)malloc(sizeof(char)*200);
        sprintf(fout,"nofile");

        while ((arg=getopt(argc,argv,"d:f:o:a:u:z:s:n:q:ih")) != -1)
        {
                switch (arg)
                {
                        case 'd':
                        if (optarg)
                        {
                                strcpy(fdata,optarg);
                                break;
                        }
                        else
                        {
                                printf("-d flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 'f':
                        if (optarg)
                        {
                                strcpy(fnam,optarg);
                                break;
                        }
                        else
                        {
                                printf("-f flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 'q':
                        if (optarg)
                        {
                                strcpy(fflag,optarg);
                                break;
                        }
                        else
                        {
                                printf("-q flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 'o':
                        if (optarg)
                        {
                                strcpy(fout,optarg);
                                break;
                        }
                        else
                        {
                                printf("-o flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 'u':
                        if (optarg)
                        {
                                nUsedAnts = atoi(optarg);
                                break;
                        }
                        else
                        {
                                printf("-u flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
						case 'z':
                        if (optarg)
                        {
                                fch1 = atof(optarg);
                                break;
                        }
                        else
                        {
                                printf("-z flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 's':
                        if (optarg)
                        {
                                sep = atof(optarg);
                                break;
                        }
                        else
                        {
                                printf("-s flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 'n':
                        if (optarg)
                        {
                                nBeamNum = atoi(optarg);
                                break;
                        }
                        else
                        {
                                printf("-n flag requires argument");
                                usage();
                                return EXIT_FAILURE;
                        }
                        case 'h':
                        usage();
                        return EXIT_SUCCESS;
                        case 'i':
			  incoh=1;
			  break;
                }
        }


        // compute beamformer weights
        //unsigned char * output = (char *)malloc(sizeof(char)*nChans*nTimes);
        unsigned char * output = (unsigned char *)malloc(sizeof(unsigned char)*nChans*nTimes);
        unsigned char * input = (char *)malloc(sizeof(char)*NANT*nChans*nTimes*nPols);
        float * antpos = (float *)malloc(sizeof(float)*NANT*2); // easting
        float * weights = (float *)malloc(sizeof(float)*NANT*NW*nPols*2); // complex weights [ant, NW, pol, r/i]
        float * wr = (float *)malloc(sizeof(float)*NANT*NW*nPols); // complex weights [ant, NW, pol]
        float * wi = (float *)malloc(sizeof(float)*NANT*NW*nPols); // complex weights [ant, NW, pol]
        float * freqs = (float *)malloc(sizeof(float)*nChans); // freq
        for (int i=0;i<nChans;i++) freqs[i] = (fch1 - i*250./8192.)*1e6;
        init_weights(fnam,fflag,antpos,weights,nPols);
        calc_weights(antpos,weights,freqs,wr,wi,sep,nBeamNum,nPols);

        FILE *ptr;
        FILE *write_ptr;
        ptr = fopen(fdata,"rb");  // r for read, b for binary
        write_ptr = fopen(fout,"wb");  // w for write, b for binary

        //long int sz;
        //fseek(ptr, 0L, SEEK_END);
        //sz = ftell(ptr);
        //rewind(ptr);
	//fseek(ptr, 369169920L, SEEK_SET);
        //int nTotSam = (int)(floor(sz / (nAnts*nChans*nTimes*nPols)));
	int nTotSam = 32768;
	
        int rd;
        for(int nSam = 0; nSam < nTotSam; nSam++) {

                rd = fread(input,NANT*nChans*nTimes*nPols,1,ptr);

                beamformer(input,wr,wi,output,nChans,nTimes,nPols,incoh);

                fwrite(output,sizeof(unsigned char),nChans*nTimes,write_ptr);

        }
        fclose(ptr);
        fclose(write_ptr);


}
