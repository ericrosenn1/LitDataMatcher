from litdatamatcher.recovery_contract import recover
def test_interrupt_corruption_pause_and_writer_guard():
 s=recover({},"TRANSIENT",writer_id="a");assert s["status"]=="RESUMABLE"
 assert recover(s,"CORRUPTION",writer_id="a")["offline_recovery"]=="CACHE_ONLY"
 assert recover(s,"USER_PAUSE",writer_id="a")["status"]=="PAUSED"
 assert recover(s,"TRANSIENT",writer_id="b")["status"]=="DUPLICATE_WRITER_BLOCKED"
