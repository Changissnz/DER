from .contextia_de_lo_vego_de_la_vegas import message_streamer
import unittest


class TestMessageStreamerClass(unittest.TestCase):


    def test_MessageStreamer__open_file(self):
        fp = "indep/text1.txt"
        fp2 = "indep/2e321b-300__EasternPoisonOak.jpg"
        fp3 = "indep/FrenchIdiomsAndProverbs__48130-0.txt"

        try:
            q = message_streamer.MessageStreamer(fp)
            q2 = message_streamer.MessageStreamer(fp2)
            q3 = message_streamer.MessageStreamer(fp3)
            assert q.mav == "tex" and q2.mav != q.mav, "incorrect file recognition"
            print("ENCODING OF ", q3.textEncoding)

            q.end_stream()
            q2.end_stream()
            q3.end_stream()
        except:
            raise ValueError("error opening files")

    """
    """
    def test_MessageStreamer__stream_block__text(self):
        fp = "indep/text1.txt"
        q = message_streamer.MessageStreamer(fp)
        q.stream_block()
        assert len(q.blockData) > 0 and len(q.blockData) <= message_streamer.DEFAULT_STREAM_BLOCK_SIZE, "invalid block data size"

        print("\n\t**DISPLAY::stream_block__text")
        for l in q.blockData:
            print(l)
        '''
        for l in q.blockData:
            print("BYTE ", l, " STR ", str(l))
        '''
        q.end_stream()

    def test_MessageStreamer__stream_block__jpg(self):
        fp = "indep/2e321b-300__EasternPoisonOak.jpg"
        q = message_streamer.MessageStreamer(fp)
        q.stream_block()
        assert len(q.blockData) == message_streamer.DEFAULT_STREAM_BLOCK_SIZE, "invalid block data size, got {}".format(len(q.blockData))

        print("\n\t**DISPLAY::stream_block__jpg")
        for l in q.blockData:
            print(l)
        q.end_stream()

    def test_MessageStreamer__stream_block__csv(self):
        fp = "indep/s.txt"
        q = message_streamer.MessageStreamer(fp,readMode="r")
        q.stream_block()

        print("\n\t**DISPLAY::stream_block__csv")
        for l in q.blockData:
            print(l)
        q.end_stream()

if __name__ == '__main__':
    unittest.main()
