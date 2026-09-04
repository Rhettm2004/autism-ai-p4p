import 'dart:typed_data';

Future<void> downloadReportPdfImpl({
  required Uint8List bytes,
  required String filename,
}) {
  throw UnsupportedError(
    'PDF download is currently supported on Flutter Web only.',
  );
}
