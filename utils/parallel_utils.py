from multiprocessing import Queue, Process, Value, cpu_count
import logging
__author__ = 'Shyam'


def run_as_parallel(worker_func, jobs_list):
    process_count = max(1, cpu_count() - 1)
    process_count = max(1, process_count)
    maxsize = 10 * process_count
    #
    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)

    worker_count = process_count
    # start worker processes
    logging.info("Using %d extract processes.", worker_count)
    workers = []
    for i in range(worker_count):
        # worker = Sampler(keep_prob=args["keep_prob"],
        #                  out_suffix=args["outext"],
        #                  outdir=args["outdir"],
        #                  thresfreq=args["thresfreq"])
        extractor = Process(target=worker_func,
                            args=(i, jobs_queue))
        extractor.daemon = True  # only live while parent process lives
        extractor.start()
        workers.append(extractor)

    # wikipath = args["wikipath"]  # "data/enwiki/enwiki_mentions_with_es-en_merged/"
    # for filename in sorted(os.listdir(wikipath)):
    #     job = (wikipath, filename)
    #     jobs_queue.put(job)  # goes to any available extract_process
    for job in jobs_list:
        jobs_queue.put(job)

    # signal termination
    for _ in workers:
        jobs_queue.put(None)
    # wait for workers to terminate
    for w in workers:
        w.join()

