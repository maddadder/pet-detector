from opentimestamps.core.timestamp import DetachedTimestampFile
from opentimestamps.core.serialize import DeserializationError
from opentimestamps.core.op import CryptOp

class CustomDetachedTimestampFile(DetachedTimestampFile):
    MAJOR_VERSION = 1  # Set to a compatible version

    def serialize(self, ctx):
        ctx.write_bytes(bytes([self.MAJOR_VERSION]))
        self.file_hash_op.serialize(ctx)
        assert self.file_hash_op.DIGEST_LENGTH == len(self.timestamp.msg)
        ctx.write_bytes(self.timestamp.msg)
        self.timestamp.serialize(ctx)

    @classmethod
    def deserialize(cls, ctx):
        ctx.assert_magic(cls.HEADER_MAGIC)
        major = ctx.read_bytes(1)  # Read the version as a byte
        if major != bytes([self.MAJOR_VERSION]):
            raise opentimestamps.core.serialize.UnsupportedMajorVersion(
                "Version %d detached timestamp files are not supported" % int.from_bytes(major, 'big')
            )
        file_hash_op = CryptOp.deserialize(ctx)
        file_hash = ctx.read_bytes(file_hash_op.DIGEST_LENGTH)
        timestamp = Timestamp.deserialize(ctx, file_hash)
        ctx.assert_eof()
        return cls(file_hash_op, timestamp)
