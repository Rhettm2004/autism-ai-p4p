import 'dart:convert';

import 'package:autism_ai/models/screening_session.dart';
import 'package:autism_ai/services/screening_session_store.dart';

class MemoryScreeningSessionStore implements ScreeningSessionStore {
  ScreeningSession? session;
  int saveCount = 0;

  @override
  Future<ScreeningSession?> load() async => session;

  @override
  Future<void> save(ScreeningSession session) async {
    final encoded = jsonEncode(session.toJson());
    final decoded = jsonDecode(encoded) as Map<String, dynamic>;
    this.session = ScreeningSession.fromJson(decoded);
    saveCount += 1;
  }

  @override
  Future<void> clear() async {
    session = null;
  }
}
