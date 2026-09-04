import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/screening_session.dart';

abstract class ScreeningSessionStore {
  Future<ScreeningSession?> load();

  Future<void> save(ScreeningSession session);

  Future<void> clear();
}

class SharedPreferencesScreeningSessionStore implements ScreeningSessionStore {
  SharedPreferencesScreeningSessionStore({SharedPreferencesAsync? preferences})
    : _preferences = preferences ?? SharedPreferencesAsync();

  static const String storageKey = 'autism_ai.screening_session.v1';

  final SharedPreferencesAsync _preferences;

  @override
  Future<ScreeningSession?> load() async {
    final encoded = await _preferences.getString(storageKey);
    if (encoded == null) return null;

    try {
      final decoded = jsonDecode(encoded);
      if (decoded is! Map) throw const FormatException('Invalid session JSON.');
      return ScreeningSession.fromJson(
        decoded.map((key, value) => MapEntry(key.toString(), value)),
      );
    } on FormatException {
      await clear();
      return null;
    } on TypeError {
      await clear();
      return null;
    }
  }

  @override
  Future<void> save(ScreeningSession session) {
    return _preferences.setString(storageKey, jsonEncode(session.toJson()));
  }

  @override
  Future<void> clear() {
    return _preferences.remove(storageKey);
  }
}

class NoopScreeningSessionStore implements ScreeningSessionStore {
  const NoopScreeningSessionStore();

  @override
  Future<ScreeningSession?> load() async => null;

  @override
  Future<void> save(ScreeningSession session) async {}

  @override
  Future<void> clear() async {}
}
