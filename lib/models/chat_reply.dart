class ChatSource {
  const ChatSource({
    required this.number,
    required this.title,
    required this.url,
    required this.authority,
    required this.lowAuthority,
    required this.passageIds,
    this.cited = false,
  });

  final int number;
  final String title;
  final String url;
  final String authority;
  final bool lowAuthority;
  final List<String> passageIds;
  final bool cited;

  Uri? get link {
    final uri = Uri.tryParse(url);
    return uri != null &&
            (uri.scheme == 'https' || uri.scheme == 'http') &&
            uri.host.isNotEmpty
        ? uri
        : null;
  }

  factory ChatSource.fromJson(Map<String, dynamic> json) {
    final number = json['number'] as int;
    final title = json['title'] as String;
    if (number < 1 || title.trim().isEmpty) {
      throw const FormatException('Invalid source');
    }
    return ChatSource(
      number: number,
      title: title,
      url: json['url'] as String,
      authority: json['authority'] as String,
      lowAuthority: json['low_authority'] as bool,
      passageIds: List<String>.unmodifiable(
        (json['passage_ids'] as List).cast<String>(),
      ),
      cited: json['cited'] as bool? ?? false,
    );
  }

  Map<String, dynamic> toJson() => {
    'number': number,
    'title': title,
    'url': url,
    'authority': authority,
    'low_authority': lowAuthority,
    'passage_ids': passageIds,
    'cited': cited,
  };
}

class ChatReply {
  const ChatReply(this.text, {this.route, this.model, this.sources = const []});
  final String text;
  final String? route;
  final String? model;
  final List<ChatSource> sources;
  // Application actions are deliberately unsupported in stages A–G.
}
