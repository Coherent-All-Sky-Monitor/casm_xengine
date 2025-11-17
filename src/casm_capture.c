/* casm_capture.c: Code to capture packets over a socket and write to a dada buffer.
This is directly taken from Vikram Ravi's dsaX_capture.c code.

cmain: runs capture loop, and interfaces dada buffer
control_thread: deals with control commands

*/
#include <arpa/inet.h>  // for ntohs and be64toh
#define __USE_GNU
#define _GNU_SOURCE
#include <sched.h>
#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <syslog.h>
#include <multilog.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/select.h>
#include <fcntl.h>
#include <errno.h>
#include <strings.h>

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "casm_capture.h"
#include "casm_def.h"
//#include "multilog.h"

#include <stdint.h>

#define CASM_F_HEADER_SIZE 16

struct casm_f_header {
    uint64_t timestamp;  // 8 bytes
    uint16_t chan0;      // 2 bytes
    uint16_t board_id;   // 2 bytes
    uint16_t n_chans;    // 2 bytes
    uint16_t n_antpols;  // 2 bytes
};

#define unhappies 3000
#define skips 6
#define sleeps 1.5

/* global variables */
int quit_threads = 0;
char STATE[20];
uint64_t UTC_START = 0; // Liam Hack... we don't have M&C yet so it will not wait for a UTC..
uint64_t UTC_STOP = 40000000000;
int MONITOR = 0;
char iP[100];
int DEBUG = 0;
int HISTOGRAM[16];
int cPort = CAPTURE_CONTROL_PORT;
int dPort = CAPTURE_PORT;

//syslog(LOG_INFO, "PSRDADA version: %s", DADA_VERSION);

void dsaX_dbgpu_cleanup (dada_hdu_t * out);
int dada_bind_thread_to_core (int core);

// Function declarations for socket diagnostics
int check_socket_status(int sock_fd);
int check_network_interface(const char* interface);
int check_socket_data_available(int sock_fd);
void print_socket_diagnostics(int sock_fd, const char* interface, int port);

// Missing type and function definitions
typedef struct {
  uint64_t total;
  uint64_t received;
  uint64_t dropped;
  uint64_t errors;
} stats_t;

stats_t* init_stats_t() {
  stats_t* stats = (stats_t*)malloc(sizeof(stats_t));
  if (stats) {
    stats->total = 0;
    stats->received = 0;
    stats->dropped = 0;
    stats->errors = 0;
  }
  return stats;
}

void reset_stats_t(stats_t* stats) {
  if (stats) {
    stats->total = 0;
    stats->received = 0;
    stats->dropped = 0;
    stats->errors = 0;
  }
}

void dsaX_dbgpu_cleanup (dada_hdu_t * out)
{

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_out");
    }
  dada_hdu_destroy (out);
  
}

void usage()
{
  fprintf (stdout,
	   "casm_capture [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -j IP to listen on for data packets [no default]\n"
	   " -p PORT to listen to for data packets [default 4011]\n"
	   " -q PORT to listen to for control commands [default CAPTURE_CONTROL_PORT]\n"
	   " -i IP to listen on for control commands [no default]\n"	
	   " -f filename of template dada header [no default]\n"
	   " -o out_key [default CAPTURE_BLOCK_KEY]\n"	   
	   " -d send debug messages to syslog\n"
	   " -h print usage\n");
}

/*
 * create a socket with the specified number of buffers
 */
dsaX_sock_t * dsaX_init_sock ()
{
  dsaX_sock_t * b = (dsaX_sock_t *) malloc(sizeof(dsaX_sock_t));
  assert(b != NULL);

  b->bufsz = sizeof(char) * UDP_PAYLOAD;

  b->buf = (char *) malloc (b->bufsz);
  assert(b->buf != NULL);

  b->have_packet = 0;
  b->fd = -1;  // Initialize to -1 so socket creation logic will run

  return b;
}

void dsaX_free_sock(dsaX_sock_t* b)
{
  b->fd = 0;
  b->bufsz = 0;
  b->have_packet =0;
  if (b->buf)
    free (b->buf);
  b->buf = 0;
}

/* 
 *  intialize UDP receiver resources
 */
int dsaX_udpdb_init_receiver (udpdb_t * ctx)
{
  syslog(LOG_INFO,"dsax_udpdb_init_receiver()");

  // create a dsaX socket which can hold variable num of UDP packet
  ctx->sock = dsaX_init_sock();

  ctx->ooo_packets = 0;
  ctx->recv_core = -1;
  ctx->n_sleeps = 0;
  ctx->mb_rcv_ps = 0;
  ctx->mb_drp_ps = 0;
  ctx->block_open = 0;
  ctx->block_count = 0;
  ctx->capture_started = 0;
  ctx->last_seq = 0;
  ctx->last_byte = 0;
  ctx->block_start_byte = 0;

  // allocate required memory strucutres
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  syslog(LOG_INFO,"receiver initiated");
  
  return 0;
}

void print_dada_header_template(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return;
    }

    printf("Contents of DADA header template file %s:\n", filename);
    printf("--------------------------------------------\n");

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        printf("%s", line);
    }

    printf("--------------------------------------------\n");

    fclose(file);
}

/* 
prepare socket and writer
*/


int dsaX_udpdb_prepare(udpdb_t * ctx)
{
  syslog(LOG_INFO, "dsaX_udpdb_prepare()");

  // Check network interface status first
  if (check_network_interface(ctx->interface) < 0) {
    syslog(LOG_WARNING, "Network interface check failed for %s, but continuing...", ctx->interface);
  }

  // open socket
  syslog(LOG_INFO, "prepare: creating udp socket on %s:%d", ctx->interface, ctx->port);
  syslog(LOG_INFO, "Attempting to create UDP socket on interface %s, port %d", ctx->interface, ctx->port);
  syslog(LOG_INFO, "dsaX_udpdb_prepare: ctx=%p, ctx->sock=%p", (void*)ctx, (void*)ctx->sock);
  ctx->verbose = 0;
  
  // Create socket if it doesn't exist or is invalid
  if (ctx->sock->fd <= 0) {
    syslog(LOG_INFO, "Creating new UDP socket...");
    ctx->sock->fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (ctx->sock->fd < 0) {
      syslog(LOG_ERR, "Failed to create socket: %s", strerror(errno));
      return -1;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(ctx->sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
      syslog(LOG_WARNING, "Failed to set SO_REUSEADDR: %s", strerror(errno));
    }
    
    // Bind socket
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ctx->port);
    
    // Handle interface binding
    if (strcmp(ctx->interface, "0.0.0.0") == 0 || strcmp(ctx->interface, "INADDR_ANY") == 0) {
      addr.sin_addr.s_addr = htonl(INADDR_ANY); // Bind to all interfaces
      syslog(LOG_INFO, "Binding to all interfaces (INADDR_ANY)");
    } else {
      // Try to bind to specific interface
      if (inet_pton(AF_INET, ctx->interface, &addr.sin_addr) != 1) {
        syslog(LOG_ERR, "Invalid interface address: %s", ctx->interface);
        close(ctx->sock->fd);
        ctx->sock->fd = -1;
        return -1;
      }
    }
    
    if (bind(ctx->sock->fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      syslog(LOG_ERR, "bind failed: %s", strerror(errno));
      close(ctx->sock->fd);
      ctx->sock->fd = -1;
      return -1;
    }
    
    syslog(LOG_INFO, "Socket created and bound successfully on FD: %d", ctx->sock->fd);
  }
  
  // set the socket size to 64 MB
  int sock_buf_size = 64*1024*1024;
  syslog(LOG_INFO, "prepare: setting buffer size to %d", sock_buf_size);
  // dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);
  syslog(LOG_INFO, "Socket created with FD: %d", ctx->sock->fd);

  // set the socket to non-blocking
  syslog(LOG_INFO, "prepare: setting non_block");
  // sock_nonblock(ctx->sock->fd);

  // clear any packets buffered by the kernel
  syslog(LOG_INFO, "prepare: clearing packets at socket");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);
  syslog(LOG_INFO, "prepare2: cleared packets at socket");

  // setup the next_seq to the initial value
  //ctx->last_seq = 0;
  //ctx->last_byte = 0;
  //ctx->n_sleeps = 0;

  return 0;
}

/*
 *  reset receiver before an observation commences
 */
void dsaX_udpdb_reset_receiver (udpdb_t * ctx) 
{
  syslog (LOG_INFO, "dsaX_udpdb_reset_receiver()");

  ctx->capture_started = 0;
  ctx->last_seq = 0;
  ctx->last_byte = 0;
  ctx->n_sleeps = 0;

  reset_stats_t(ctx->packets);
  reset_stats_t(ctx->bytes);
}

/* 
 *  open a data block buffer ready for direct access
 */
int dsaX_udpdb_open_buffer (udpdb_t * ctx)
{

  if (DEBUG) syslog (LOG_DEBUG, "dsaX_udpdb_open_buffer()");

  if (ctx->block_open)
  {
    syslog (LOG_ERR, "open_buffer: buffer already opened");
    return -1;
  }

  if (DEBUG) syslog (LOG_DEBUG, "open_buffer: ipcio_open_block_write");

  uint64_t block_id = 0;

  ctx->block = ipcio_open_block_write (ctx->hdu->data_block, &block_id);
  if (!ctx->block)
  { 
    syslog (LOG_ERR, "open_buffer: ipcio_open_block_write failed");
    return -1;
  }

  ctx->block_open = 1;
  ctx->block_count = 0;

  return 0;
}

/*
 *  close a data buffer, assuming a full block has been written
 */
int dsaX_udpdb_close_buffer (udpdb_t * ctx, uint64_t bytes_written, unsigned eod)
{

  if (DEBUG) syslog (LOG_DEBUG, "dsaX_udpdb_close_buffer(%"PRIu64", %d)", bytes_written, eod);

  if (!ctx->block_open)
  { 
    syslog (LOG_ERR, "close_buffer: buffer already closed");
    return -1;
  }

  // log any buffers that are not full, except for the 1 byte "EOD" buffer
  if ((bytes_written != 1) && (bytes_written != ctx->hdu_bufsz))
    syslog ((eod ? LOG_INFO : LOG_WARNING), "close_buffer: "
              "bytes_written[%"PRIu64"] != hdu_bufsz[%"PRIu64"]", 
              bytes_written, ctx->hdu_bufsz);

  if (eod)
  {
    if (ipcio_update_block_write (ctx->hdu->data_block, bytes_written) < 0)
    {
      syslog (LOG_ERR, "close_buffer: ipcio_update_block_write failed");
      return -1;
    }
  }
  else 
  {
    if (ipcio_close_block_write (ctx->hdu->data_block, bytes_written) < 0)
    {
      syslog (LOG_ERR, "close_buffer: ipcio_close_block_write failed");
      return -1;
    }
  }

  ctx->block = 0;
  ctx->block_open = 0;

  return 0;
}

/* 
 *  move to the next ring buffer element. return pointer to base address of new buffer
 */
int dsaX_udpdb_new_buffer (udpdb_t * ctx)
{

  if (DEBUG) syslog (LOG_DEBUG, "dsaX_udpdb_new_buffer()");

  if (dsaX_udpdb_close_buffer (ctx, ctx->hdu_bufsz, 0) < 0)
  {
    syslog (LOG_ERR, "new_buffer: dsaX_udpdb_close_buffer failed");
    return -1;
  }

  if (dsaX_udpdb_open_buffer (ctx) < 0) 
  {
    syslog (LOG_ERR, "new_buffer: dsaX_udpdb_open_buffer failed");
    return -1;
  }

  // increment buffer byte markers
  ctx->block_start_byte = ctx->block_end_byte + UDP_DATA;
  ctx->block_end_byte = ctx->block_start_byte + ( ctx->packets_per_buffer - 1) * UDP_DATA;

  // set block to 0
  //memset(ctx->block,0,ctx->block_end_byte-ctx->block_start_byte);
  
  if (DEBUG) syslog(LOG_DEBUG, "new_buffer: buffer_bytes [%"PRIu64" - %"PRIu64"]", 
             ctx->block_start_byte, ctx->block_end_byte);

  return 0;

}

/* 
 *  destroy UDP receiver resources 
 */
int dsaX_udpdb_destroy_receiver (udpdb_t * ctx)
{
  if (ctx->sock)
    dsaX_free_sock(ctx->sock);
  ctx->sock = 0;
  return 0;
}

/*
 * Close the udp socket and file
 */

int udpdb_stop_function (udpdb_t* ctx)
{

  syslog(LOG_INFO, "stop: dada_hdu_unlock_write()");
  if (dada_hdu_unlock_write (ctx->hdu) < 0)
  {
    syslog (LOG_ERR, "stop: could not unlock write on");
    return -1;
  }

  // close the UDP socket
  close(ctx->sock->fd);

  if (ctx->packets->dropped)
  {
    double percent = (double) ctx->bytes->dropped / (double) ctx->last_byte;
    percent *= 100;

  }

  return 0;
}


/* --------- THREADS -------- */

// STATS THREAD

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {

  /*  // set affinity
  const pthread_t pid = pthread_self();
  const int core_id = 4;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (set_result != 0)
    syslog(LOG_ERR,"thread %d: setaffinity_np fail",core_id);
  const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (get_affinity != 0) 
    syslog(LOG_ERR,"thread %d: getaffinity_np fail",core_id);
  if (CPU_ISSET(core_id, &cpuset))
    syslog(LOG_INFO,"thread %d: successfully set thread",core_id);
  */
  
  udpdb_t * ctx = (udpdb_t *) arg;
  uint64_t b_rcv_total = 0;
  uint64_t b_rcv_1sec = 0;
  uint64_t b_rcv_curr = 0;

  uint64_t b_drp_total = 0;
  uint64_t b_drp_1sec = 0;
  uint64_t b_drp_curr = 0;

  uint64_t s_rcv_total = 0;
  uint64_t s_rcv_1sec = 0;
  uint64_t s_rcv_curr = 0;

  uint64_t ooo_pkts = 0;
  float gb_rcv_ps = 0;
  float mb_rcv_ps = 0;
  float mb_drp_ps = 0;

  syslog(LOG_INFO,"stats_thread: starting loop");
  
  while (!quit_threads)
  {

    /* get a snapshot of the data as quickly as possible */

    b_rcv_curr = ctx->bytes->received;
    b_drp_curr = ctx->bytes->dropped;
    s_rcv_curr = ctx->n_sleeps;
    
    /* calc the values for the last second */
    b_rcv_1sec = b_rcv_curr - b_rcv_total;
    b_drp_1sec = b_drp_curr - b_drp_total;
    s_rcv_1sec = s_rcv_curr - s_rcv_total;

    /* update the totals */
    b_rcv_total = b_rcv_curr;
    b_drp_total = b_drp_curr;
    s_rcv_total = s_rcv_curr;

    mb_rcv_ps = (double) b_rcv_1sec / 1000000;
    mb_drp_ps = (double) b_drp_1sec / 1000000;
    gb_rcv_ps = b_rcv_1sec * 8;
    gb_rcv_ps /= 1000000000;

    /* determine how much memory is free in the receivers */
    //    syslog (LOG_NOTICE,"CAPSTATS %6.3f [Gb/s], D %4.1f [MB/s], D %"PRIu64" pkts, %"PRIu64" skipped 0", gb_rcv_ps, mb_drp_ps, ctx->packets->dropped, ctx->last_seq);

    sleep(1);
  }

}


typedef struct {
    uint64_t timestamp;
    uint16_t chan0;
    uint16_t board_id;
    uint16_t n_chans;
    uint16_t n_antpols;
} PacketDetails;

PacketDetails extract_packet_details(const unsigned char* buffer, size_t size) {
    PacketDetails details = {0};

    if (size < 16) {
        // Handle error: packet too small
        return details;
    }

    memcpy(&details.timestamp, buffer, sizeof(uint64_t));
    memcpy(&details.chan0, buffer + 8, sizeof(uint16_t));
    memcpy(&details.board_id, buffer + 10, sizeof(uint16_t));
    memcpy(&details.n_chans, buffer + 12, sizeof(uint16_t));
    memcpy(&details.n_antpols, buffer + 14, sizeof(uint16_t));

    // Convert from network byte order to host byte order
    details.timestamp = be64toh(details.timestamp);
    details.chan0 = ntohs(details.chan0);
    details.board_id = ntohs(details.board_id);
    details.n_chans = ntohs(details.n_chans);
    details.n_antpols = ntohs(details.n_antpols);

    return details;
}




// CONTROL THREAD

void control_thread (void * arg) {

  udpdb_t * ctx = (udpdb_t *) arg;
  syslog(LOG_INFO, "control_thread: starting");

  // port on which to listen for control commands
  int port = cPort;
  char sport[10];
  sprintf(sport,"%d",port);

  // buffer for incoming command strings, and setup of socket
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  memset(buffer, '\0', bufsize);
  const char* whitespace = " ";
  char * command = 0;
  char * args = 0;

  struct addrinfo hints;
  struct addrinfo* res=0;
  memset(&hints,0,sizeof(hints));
  int data_sockfd;
  struct sockaddr_storage src_addr;
  
  socklen_t src_addr_len=sizeof(src_addr);
  hints.ai_family=AF_INET;
  hints.ai_socktype=SOCK_DGRAM;
  getaddrinfo(iP,sport,&hints,&res);
  int fd;
  ssize_t ct;
  char tmpstr;
  char cmpstr = 'p';
  char *endptr;
  uint64_t tmps;
  char * token;
  
  syslog(LOG_INFO, "control_thread: created socket on port %d", port);
  
  while (!quit_threads) {
    
    fd = socket(res->ai_family,res->ai_socktype,res->ai_protocol);
    bind(fd,res->ai_addr,res->ai_addrlen);
    memset(buffer,'\0',sizeof(buffer));
    syslog(LOG_INFO, "control_thread: waiting for packet");
    ct = recvfrom(fd,buffer,1024,0,(struct sockaddr*)&src_addr,&src_addr_len);
    
    syslog(LOG_INFO, "control_thread: received buffer string %s",buffer);

    // INTERPRET BUFFER STRING
    // receive either UTC_START, UTC_STOP, MONITOR

    // interpret buffer string
    char * rest = buffer;
    char *cmd, *val;
    cmd = strtok_r(rest, "-", &rest);
    val = strtok_r(rest, "-", &rest);
    syslog(LOG_INFO, "control_thread: split into COMMAND %s, VALUE %s",cmd,val);

    if (strcmp(cmd,"UTC_START")==0)
      UTC_START = strtoull(val,&endptr,0);

    if (strcmp(cmd,"UTC_STOP")==0)
      UTC_STOP = strtoull(val,&endptr,0);    
    
    close(fd);
    
  }

  free (buffer);

  syslog(LOG_INFO, "control_thread: exiting");

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}


void log_packet_details(const unsigned char* buffer, size_t size) {
    if (size < 16) {
        syslog(LOG_WARNING, "Packet too small to contain full header");
        return;
    }

    uint64_t timestamp;
    uint16_t chan0, board_id, n_chans, n_antpols;

    memcpy(&timestamp, buffer, sizeof(uint64_t));
    memcpy(&chan0, buffer + 8, sizeof(uint16_t));
    memcpy(&board_id, buffer + 10, sizeof(uint16_t));
    memcpy(&n_chans, buffer + 12, sizeof(uint16_t));
    memcpy(&n_antpols, buffer + 14, sizeof(uint16_t));

    syslog(LOG_INFO, "Packet Details: size=%zu, timestamp=%lu, chan0=%u, board_id=%u, n_chans=%u, n_antpols=%u",
           size, be64toh(timestamp), ntohs(chan0), ntohs(board_id), ntohs(n_chans), ntohs(n_antpols));
}

/*
 * Check if socket is still valid and healthy
 */
int check_socket_status(int sock_fd) {
    if (sock_fd < 0) {
        syslog(LOG_ERR, "Socket file descriptor is invalid: %d", sock_fd);
        return -1;
    }
    
    // Check if socket is still valid
    int error = 0;
    socklen_t len = sizeof(error);
    int retval = getsockopt(sock_fd, SOL_SOCKET, SO_ERROR, &error, &len);
    
    if (retval != 0) {
        syslog(LOG_ERR, "getsockopt failed: %s", strerror(errno));
        return -1;
    }
    
    if (error != 0) {
        syslog(LOG_ERR, "Socket error: %s", strerror(error));
        return -1;
    }
    
    // Check socket buffer info
    int recv_buf_size = 0;
    socklen_t optlen = sizeof(recv_buf_size);
    if (getsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &recv_buf_size, &optlen) == 0) {
        syslog(LOG_INFO, "Socket receive buffer size: %d bytes", recv_buf_size);
    }
    
    syslog(LOG_INFO, "Socket status: OK");
    return 0;
}

/*
 * Check if network interface exists and is working
 */
int check_network_interface(const char* interface) {
    struct ifreq ifr;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    
    if (sock < 0) {
        syslog(LOG_ERR, "Cannot create socket for interface check");
        return -1;
    }
    
    strncpy(ifr.ifr_name, interface, IFNAMSIZ-1);
    
    // Check if interface exists and is up
    if (ioctl(sock, SIOCGIFFLAGS, &ifr) < 0) {
        syslog(LOG_ERR, "Interface %s not found", interface);
        close(sock);
        return -1;
    }
    
    if (!(ifr.ifr_flags & IFF_UP)) {
        syslog(LOG_ERR, "Interface %s is down", interface);
        close(sock);
        return -1;
    }
    
    if (!(ifr.ifr_flags & IFF_RUNNING)) {
        syslog(LOG_ERR, "Interface %s is not running", interface);
        close(sock);
        return -1;
    }
    
    syslog(LOG_INFO, "Interface %s is up and running", interface);
    close(sock);
    return 0;
}

/*
 * Check if there's any data available on the socket
 */
int check_socket_data_available(int sock_fd) {
    if (sock_fd < 0) {
        return -1;
    }
    
    fd_set read_fds;
    struct timeval timeout;
    
    FD_ZERO(&read_fds);
    FD_SET(sock_fd, &read_fds);
    
    // Set timeout to 0 for non-blocking check
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;
    
    int result = select(sock_fd + 1, &read_fds, NULL, NULL, &timeout);
    
    if (result < 0) {
        syslog(LOG_ERR, "select() failed: %s", strerror(errno));
        return -1;
    } else if (result == 0) {
        syslog(LOG_INFO, "No data available on socket (timeout)");
        return 0;
    } else {
        if (FD_ISSET(sock_fd, &read_fds)) {
            syslog(LOG_INFO, "Data is available on socket");
            return 1;
        }
    }
    
    return 0;
}

/*
 * Comprehensive socket diagnostic function
 */
void print_socket_diagnostics(int sock_fd, const char* interface, int port) {
    syslog(LOG_INFO, "=== SOCKET DIAGNOSTICS ===");
    
    // Basic socket info
    if (sock_fd < 0) {
        syslog(LOG_ERR, "Socket file descriptor: INVALID (%d)", sock_fd);
        return;
    } else {
        syslog(LOG_INFO, "Socket file descriptor: %d", sock_fd);
    }
    
    // Check socket status
    if (check_socket_status(sock_fd) < 0) {
        syslog(LOG_ERR, "Socket health check: FAILED");
    } else {
        syslog(LOG_INFO, "Socket health check: PASSED");
    }
    
    // Check network interface
    if (check_network_interface(interface) < 0) {
        syslog(LOG_ERR, "Network interface check: FAILED");
    } else {
        syslog(LOG_INFO, "Network interface check: PASSED");
    }
    
    // Check if data is available
    int data_status = check_socket_data_available(sock_fd);
    if (data_status < 0) {
        syslog(LOG_ERR, "Data availability check: FAILED");
    } else if (data_status == 0) {
        syslog(LOG_INFO, "Data availability check: NO DATA");
    } else {
        syslog(LOG_INFO, "Data availability check: DATA AVAILABLE");
    }
    
    // Socket options
    int recv_buf_size = 0;
    socklen_t optlen = sizeof(recv_buf_size);
    if (getsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &recv_buf_size, &optlen) == 0) {
        syslog(LOG_INFO, "Receive buffer size: %d bytes", recv_buf_size);
    }
    
    int send_buf_size = 0;
    optlen = sizeof(send_buf_size);
    if (getsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &send_buf_size, &optlen) == 0) {
        syslog(LOG_INFO, "Send buffer size: %d bytes", send_buf_size);
    }
    
    // Check if socket is non-blocking
    int flags = fcntl(sock_fd, F_GETFL, 0);
    if (flags < 0) {
        syslog(LOG_ERR, "Cannot get socket flags");
    } else {
        if (flags & O_NONBLOCK) {
            syslog(LOG_INFO, "Socket mode: NON-BLOCKING");
        } else {
            syslog(LOG_INFO, "Socket mode: BLOCKING");
        }
    }
    
    syslog(LOG_INFO, "=== END DIAGNOSTICS ===");
}

// MAIN of program
	
int main (int argc, char *argv[]) {


  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_capture", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit for writing */
  dada_hdu_t* hdu_out = 0;

  /* actual struct with info */
  udpdb_t udpdb;
  memset(&udpdb, 0, sizeof(udpdb));

  // input data block HDU key
  key_t out_key = CAPTURE_BLOCK_KEY;

  // command line arguments
  int core = -1;
  int arg=0;
  char dada_fnam[200]; // filename for dada header
  char iface[100]; // IP for data packets
  
  while ((arg=getopt(argc,argv,"c:j:i:f:o:g:p:q:dh")) != -1)
    {
      switch (arg)
	{
	case 'o':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &out_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-o flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'i':
	  if (optarg)
	    {	      
	      strcpy(iP,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-i flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'p':
	  if (optarg)
	    {	      
	      dPort = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-p flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'q':
	  if (optarg)
	    {	      
	      cPort = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-q flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'j':
	  if (optarg)
	    {	      
	      strcpy(iface,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-j flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'c':
	  if (optarg)
	    {
	      core = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-c flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }      	
	case 'f':
	  if (optarg)
	    {	      
	      strcpy(dada_fnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }	 
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // record STATE info
  sprintf(STATE,"NOBUFFER");
  //  print_dada_header_template(dada_fnam);
  // START THREADS
  
  // initialize the data structure BEFORE starting threads that read it
  syslog (LOG_INFO, "main: dsaX_udpdb_init_receiver()");
  if (dsaX_udpdb_init_receiver (&udpdb) < 0)
  {
    syslog (LOG_ERR, "could not initialize receiver");
    return EXIT_FAILURE;
  }
  
  // start control thread
  int rval = 0;
  pthread_t control_thread_id, stats_thread_id;
  if (DEBUG)
    syslog (LOG_DEBUG, "Creating threads");
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpdb);
  if (rval != 0) {
    syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }
  syslog(LOG_NOTICE, "Created control thread, listening on %s:%d",iP,cPort);

  // start the stats thread
  rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udpdb);
  if (rval != 0) {
    syslog(LOG_INFO, "Error creating stats_thread: %s", strerror(rval));
    return -1;
  }
  syslog(LOG_NOTICE, "started stats_thread()");

  
  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }

  // OPEN CONNECTION TO DADA DB FOR WRITING

  if (DEBUG) syslog(LOG_INFO,"Creating HDU");
  
  multilog_t* log = multilog_open("dsaX_capture", 0);
  hdu_out = dada_hdu_create(log);
  if (DEBUG) syslog(LOG_INFO,"Created hdu");
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog(LOG_ERR,"could not connect to output dada buffer");
    return EXIT_FAILURE;
  }
  if (DEBUG) syslog(LOG_INFO,"Connected HDU");
  if (dada_hdu_lock_write(hdu_out) < 0) {
    dsaX_dbgpu_cleanup (hdu_out);
    syslog(LOG_ERR,"could not lock to output dada buffer");
    return EXIT_FAILURE;
  }

  syslog(LOG_INFO,"opened connection to output DB");

  // DEAL WITH DADA HEADER
  char *hout;
  hout = (char *)malloc(sizeof(char)*4096);
  if (DEBUG) syslog(LOG_INFO,"read header2");

  
  if (fileread (dada_fnam, hout, 4096) < 0)
    {
      free (hout);
      syslog (LOG_ERR, "could not read ASCII header from %s", dada_fnam);
      return (EXIT_FAILURE);
    }
  
  if (DEBUG) syslog(LOG_INFO,"read header3");  

  syslog(LOG_DEBUG, "About to call ipcbuf_get_next_write");
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  syslog(LOG_DEBUG, "Returned from ipcbuf_get_next_write");

  if (!header_out)
    {
      syslog(LOG_DEBUG, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_out);
      return EXIT_FAILURE;
    }


  // copy the in header to the out header
  memcpy (header_out, hout, 4096);

  // mark the output header buffer as filled
  if (ipcbuf_mark_filled (hdu_out->header_block, 4096) < 0)
    {
      syslog(LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_out);
      return EXIT_FAILURE;
    }

  // record STATE info
  sprintf(STATE,"LISTEN");
  syslog(LOG_INFO,"marked output header block as filled - now in LISTEN state");


  /* time to start up receiver. 
     data are captured on iface:CAPTURE_PORT 
  */

  // put information in udpdb struct
  udpdb.hdu = hdu_out;
  udpdb.port = dPort;
  udpdb.interface = strdup(iface);
  udpdb.hdu_bufsz = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  udpdb.packets_per_buffer = udpdb.hdu_bufsz / UDP_DATA;
  udpdb.num_inputs = NSNAPS;
  udpdb.verbose = 0;

  // provide valid logger to PSRDADA helpers
  udpdb.log = log; 

  // determine number of packets per block, must 
  if (udpdb.hdu_bufsz % UDP_DATA != 0)
  {
    syslog(LOG_ERR, "data block size for [%"PRIu64"] was not a multiple of the UDP_DATA size [%d]\n", udpdb.hdu_bufsz, UDP_DATA);
    return EXIT_FAILURE;
  }
  udpdb.packets_per_buffer = udpdb.hdu_bufsz / UDP_DATA;  
  udpdb.bytes_to_acquire = 0;
  udpdb.num_inputs = NSNAPS;

  // prepare the socket
  syslog(LOG_INFO, "main: dsaX_udpdb_prepare()");
  if (dsaX_udpdb_prepare (&udpdb) < 0)
  {
    syslog(LOG_ERR, "could allocate required resources (prepare)");
    return EXIT_FAILURE;
  }
  
  // Run comprehensive socket diagnostics after preparation
  syslog(LOG_INFO, "Running initial socket diagnostics...");
  print_socket_diagnostics(udpdb.sock->fd, udpdb.interface, udpdb.port);
  
  syslog(LOG_INFO, "Test");  
  // reset the receiver
  syslog(LOG_INFO, "main: dsaX_udpdb_reset_receiver()");
  dsaX_udpdb_reset_receiver (&udpdb);

  // open a block of the data block, ready for writing
  if (dsaX_udpdb_open_buffer (&udpdb) < 0)
  {
    syslog (LOG_ERR, "start: dsaX_udpdb_open_buffer failed");
    return -1;
  }
  
  /* START WHAT WAS in RECV THREAD */

  // DEFINITIONS
  syslog(LOG_INFO, "definitions");

  // lookup table for ant order
  uint64_t ant_lookup[100], vv;
  for (int i=0;i<100;i++) ant_lookup[i] = 0;
  for (int i=0;i<NSNAPS/2;i++) {
    for (int j=0;j<2;j++) {
      vv = (i*2+j)*3;
      ant_lookup[vv] = (uint64_t)(i);
    }
  }


  int unhappies_ct = 0;
  int unhappy = 0;
  uint64_t act_seq_no = 0;
  uint64_t block_seq_no = 0;
  uint64_t seq_no = 0;
  uint64_t ch_id = 0;
  uint64_t ant_id = 0, aid;
  unsigned char * b = (unsigned char *) udpdb.sock->buf;
  size_t got = 0; // data received from a recv_from call
  int errsv; // determine the sequence number boundaries for curr and next buffers
  int64_t byte_offset = 0; // offset of current packet in bytes from start of block
  uint64_t seq_byte = 0; // offset of current packet in bytes from start of obs
  // for "saving" out of order packets near edges of blocks
  unsigned int temp_idx = 0;
  unsigned int temp_max = 1000;
  char ** temp_buffers; //[temp_max][UDP_DATA];
  uint64_t * temp_seq_byte;
  temp_buffers = (char **)malloc(sizeof(char *)*temp_max);
  for (int i=0;i<temp_max;i++) temp_buffers[i] = (char *)malloc(sizeof(char)*UDP_DATA);
  temp_seq_byte = (uint64_t *)malloc(sizeof(uint64_t)*temp_max);
  unsigned i = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000000;
  int canWrite = 0;
  int ct_snaps=0;

  // infinite loop to receive packets
  // use stats thread to monitor STATE at this stage, to save resources here

  syslog(LOG_INFO, "Starting infinite loop");
  
  // Socket health check counter
  static int check_counter = 0;
  
  while (1)
    {
      // Periodic socket health check every 1000 iterations
      check_counter++;
      if (check_counter % 1000 == 0) {
        if (check_socket_status(udpdb.sock->fd) < 0) {
          syslog(LOG_ERR, "Socket health check failed, running diagnostics...");
          print_socket_diagnostics(udpdb.sock->fd, udpdb.interface, udpdb.port);
          
          syslog(LOG_ERR, "Re-preparing socket...");
          // Re-prepare the socket
          if (dsaX_udpdb_prepare(&udpdb) < 0) {
            syslog(LOG_ERR, "Failed to re-prepare socket");
            return EXIT_FAILURE;
          }
          
          // Run diagnostics again after re-preparation
          syslog(LOG_INFO, "Running diagnostics after socket re-preparation...");
          print_socket_diagnostics(udpdb.sock->fd, udpdb.interface, udpdb.port);
        }
      }

      udpdb.sock->have_packet = 0; 

      // incredibly tight loop to try and get a packet
      while (!udpdb.sock->have_packet)
	{
	 
	  // receive 1 packet into the socket buffer
	  got = recvfrom ( udpdb.sock->fd, udpdb.sock->buf, UDP_PAYLOAD, 0, NULL, NULL );

	  //log_packet_details(udpdb.sock->buf, got);
	  syslog(LOG_INFO, "Received packet of size %zd", got);
	  
	  if (got == UDP_PAYLOAD) 
	    {
	      udpdb.sock->have_packet = 1;
	    } 
	  else if (got == -1) 
	    {
	      errsv = errno;
	      if (errsv == EAGAIN) 
		{
		  udpdb.n_sleeps++;
		  if (udpdb.capture_started)
		    timeouts++;
		  if (timeouts > timeout_max)
		    syslog(LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);		  
		}
	      else 
		{
		  syslog (LOG_ERR, "receive_obs: recvfrom failed %s", strerror(errsv));
		  // Log additional socket diagnostics
		  check_socket_status(udpdb.sock->fd);
		  return EXIT_FAILURE;
		}
	    } 
	  else // we received a packet of the WRONG size, ignore it
	    {
	      syslog(LOG_NOTICE, "receive_obs: received %zu bytes, expected %d", got, UDP_PAYLOAD);
	    }
	}
      timeouts = 0;

      // we have a valid packet within the timeout
      if (udpdb.sock->have_packet) 
	{
	  PacketDetails details = extract_packet_details(udpdb.sock->buf, got);

	  // Now you can use the extracted values
	  uint64_t seq_no = details.timestamp;
	  uint16_t aid = details.chan0/512;  // This will give values 0,1,2,3,4,5
	  
	  if (UTC_START == 0) UTC_START = details.timestamp;

	  // You can also log the details if needed
	  
	  if (UTC_START==0) UTC_START = seq_no;
	  
	  //act_seq_no = seq_no*NCHANG*NSNAPS/2 + ant_id*NCHANG/3 + (ch_id-CHOFF)/384; // actual seq no
	  act_seq_no = NCHANG*seq_no + aid; // actual seq no
	  block_seq_no = NCHANG*UTC_START; // seq no corresponding to ant 0 and start of block

	  //syslog(LOG_INFO, "Packet Details: act_seq_no=%lu, block_seq_no=%lu, seq_no=%lu, n_chans=%u, n_antpols=%u",

	  //	 act_seq_no, block_seq_no, seq_no, details.n_chans, aid);	  	  

	  
	  // check for starting or stopping condition, using continue
	  //if (DEBUG) printf("%"PRIu64" %"PRIu64" %d\n",seq_no,act_seq_no,ch_id);//syslog(LOG_DEBUG, "seq_byte=%"PRIu64", num_inputs=%d, seq_no=%"PRIu64", ant_id =%"PRIu64", ch_id =%"PRIu64"",seq_byte,udpdb.num_inputs,seq_no,ant_id, ch_id);
	  //if (seq_no == UTC_START && UTC_START != 10000 && ant_id == 0) canWrite=1;
	  //	  syslog(LOG_INFO, "seq_no: %"PRIu64", UTC_START: %"PRIu64", ct_snaps: %d, canWrite: %d", seq_no, UTC_START, ct_snaps, canWrite); Liam Hack
	  //if (canWrite==0) {
	  //  if (seq_no >= UTC_START-5 && UTC_START != 10000) ct_snaps++;
	  //  if (ct_snaps >= 32) canWrite=1;
	  //}
	  // check for starting or stopping condition, using continue
	  if (canWrite==0) {
	    if (seq_no >= UTC_START) {
	      canWrite=1;	      
	    }
	  }
	  if (canWrite == 0) continue;

	  //if (seq_no > UTC_START && UTC_START != 10000) canWrite=1;	  
	  udpdb.last_seq = seq_no;
	  //	  syslog(LOG_INFO, "seq_no: %"PRIu64", UTC_START: %"PRIu64", ct_snaps: %d, canWrite: %d", seq_no, UTC_START, ct_snaps, canWrite);
	  //if (seq_no == UTC_STOP) canWrite=0;
	  //if (udpdb.packets->received<100) syslog(LOG_INFO, "seq_byte=%"PRIu64", num_inputs=%d, seq_no=%"PRIu64", ant_id =%"PRIu64", ch_id =%"PRIu64"",seq_byte,udpdb.num_inputs,seq_no,ant_id, ch_id);
	  //	  syslog(LOG_INFO, "%d", udpdb.capture_started);
	  // if first packet
	  if (!udpdb.capture_started)
	    {
	      //udpdb.block_start_byte = act_seq_no * UDP_DATA;
	      udpdb.block_start_byte = block_seq_no * UDP_DATA;
	      udpdb.block_end_byte   = (udpdb.block_start_byte + udpdb.hdu_bufsz) - UDP_DATA;
	      udpdb.capture_started = 1;


	      syslog (LOG_INFO, "receive_obs: START [%"PRIu64" - %"PRIu64"]", udpdb.block_start_byte, udpdb.block_end_byte);
	    }

	  //syslog(LOG_INFO, "Packet belongs in this block: act_seq_no=%lu, block_seq_no=%lu, seq_no=%lu",
	  //		 act_seq_no, block_seq_no, seq_no);	  	  


	  // if capture running
	  if (udpdb.capture_started)
	    {
	      seq_byte = (act_seq_no * UDP_DATA);	      

	      udpdb.last_byte = seq_byte;
	      
	      // if packet arrived too late, ignore
	      if (seq_byte < udpdb.block_start_byte)
		{
		  syslog (LOG_INFO, "receive_obs: seq_byte < block_start_byte: %"PRIu64"", seq_no);
		  udpdb.packets->dropped++;
		  udpdb.bytes->dropped += UDP_DATA;
		}
	      else
		{
		  // packet belongs in this block
		  if (seq_byte <= udpdb.block_end_byte)
		    {
		      byte_offset = seq_byte - udpdb.block_start_byte;
		      memcpy (udpdb.block + byte_offset, udpdb.sock->buf + UDP_HEADER, UDP_DATA);
		      udpdb.packets->received++;
		      udpdb.bytes->received += UDP_DATA;
		      udpdb.block_count++;
		      syslog (LOG_INFO, "RECEIVED UDP_HEADERSIZE=%d", UDP_HEADER);
		    }
		      //		      syslog (LOG_INFO, "Received");
		  
		  // packet belongs in subsequent block
		  else
		    {
		      //syslog (LOG_INFO, "receive_obs: received packet for subsequent buffer: temp_idx=%d, ant_id=%d, seq_no=%"PRIu64"",temp_idx,ant_id,seq_no);
		      
		      if (temp_idx < temp_max)
			{
			  // save packet to temp buffer
			  memcpy (temp_buffers[temp_idx], udpdb.sock->buf + UDP_HEADER, UDP_DATA);
			  temp_seq_byte[temp_idx] = seq_byte;
			  temp_idx++;
			}
		      else
			{
			  udpdb.packets->dropped++;
			  udpdb.bytes->dropped += UDP_DATA;
			}
		    }
		}
	    }

	  // now check for a full buffer or full temp queue
	  if ((udpdb.block_count >= udpdb.packets_per_buffer) || (temp_idx >= temp_max))
	    {
	      //	      syslog (LOG_INFO, "BLOCK COMPLETE seq_no=%"PRIu64", "
	      //	      "ant_id=%"PRIu16", block_count=%"PRIu64", "
	      //      "temp_idx=%d\n", seq_no, ant_id,  udpdb.block_count, 
	      //      temp_idx);
	      
	      uint64_t dropped = udpdb.packets_per_buffer - udpdb.block_count;
	      if (dropped)
		{
		  udpdb.packets->dropped += dropped;
		  udpdb.bytes->dropped += (dropped * UDP_DATA);
		}

	      if (dropped>1000) unhappies_ct++;

	      // get a new buffer and write any temp packets saved 
	      if (dsaX_udpdb_new_buffer (&udpdb) < 0)
		{
		  syslog(LOG_ERR, "receive_obs: dsaX_udpdb_new_buffer failed");
		  return EXIT_FAILURE;
		}

	      if (DEBUG) syslog(LOG_INFO, "block bytes: %"PRIu64" - %"PRIu64"\n", udpdb.block_start_byte, udpdb.block_end_byte);
  
	      // include any futuristic packets we saved
	      for (i=0; i < temp_idx; i++)
		{
		  seq_byte = temp_seq_byte[i];
		  byte_offset = seq_byte - udpdb.block_start_byte;
		  if (byte_offset < udpdb.hdu_bufsz)
		    {
		      memcpy (udpdb.block + byte_offset, temp_buffers[i], UDP_DATA);
		      udpdb.block_count++;
		      udpdb.packets->received++;
		      udpdb.bytes->received += UDP_DATA;
		    }
		  else
		    {
		      udpdb.packets->dropped++;
		      udpdb.bytes->dropped += UDP_DATA;
		    }
		}
	      temp_idx = 0;
	    }
	}

      // packet has been inserted or saved by this point
      udpdb.sock->have_packet = 0;

      // deal with unhappy receiver
      if (unhappies_ct > unhappies) {

	syslog(LOG_INFO, "Skipping some blocks...");

	close(udpdb.sock->fd);

	for (int i=0;i<skips;i++) {

	  udpdb.packets->dropped += udpdb.packets_per_buffer;
	  udpdb.bytes->dropped += (udpdb.packets_per_buffer * UDP_DATA);

	  if (dsaX_udpdb_new_buffer (&udpdb) < 0)
	    {
	      syslog(LOG_ERR, "receive_obs: dsaX_udpdb_new_buffer failed");
	      return EXIT_FAILURE;
	    }

	}

	sleep(sleeps);

	// prepare the socket
	syslog(LOG_INFO, "re-preparing the socket dsaX_udpdb_prepare()");
	if (dsaX_udpdb_prepare (&udpdb) < 0)
	  {
	    syslog(LOG_ERR, "could allocate required resources (prepare)");
	    return EXIT_FAILURE;
	  }	
	
	unhappies_ct = 0;

      }
      
    }

  /* END WHAT WAS IN RECV THREAD */
  

  // close threads
  syslog(LOG_INFO, "joining control_thread and stats_thread");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);
  pthread_join (stats_thread_id, &result);

  free(temp_seq_byte);
  free(temp_buffers);
  
  dsaX_dbgpu_cleanup (hdu_out);

}
