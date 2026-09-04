import 'package:flutter/material.dart';

import '../app.dart';
import '../models/screening_models.dart';

class PersistentChatPanel extends StatefulWidget {
  const PersistentChatPanel({
    super.key,
    required this.messages,
    required this.onSend,
    required this.enabled,
    required this.isSending,
    required this.inputFocusNode,
  });

  final List<ChatMessage> messages;
  final Future<void> Function(String) onSend;
  final bool enabled;
  final bool isSending;
  final FocusNode inputFocusNode;

  @override
  State<PersistentChatPanel> createState() => _PersistentChatPanelState();
}

class _PersistentChatPanelState extends State<PersistentChatPanel> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void didUpdateWidget(covariant PersistentChatPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.messages.length != widget.messages.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToEnd());
    }
  }

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final text = _messageController.text.trim();
    if (text.isEmpty || !widget.enabled || widget.isSending) return;
    _messageController.clear();
    await widget.onSend(text);
  }

  void _scrollToEnd() {
    if (!_scrollController.hasClients) return;
    _scrollController.animateTo(
      _scrollController.position.maxScrollExtent,
      duration: const Duration(milliseconds: 220),
      curve: Curves.easeOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          Container(
            height: 38,
            padding: const EdgeInsets.symmetric(horizontal: 14),
            decoration: const BoxDecoration(
              color: AppColors.softBlue,
              border: Border(bottom: BorderSide(color: AppColors.border)),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.smart_toy_outlined,
                  size: 19,
                  color: AppColors.blue,
                ),
                const SizedBox(width: 8),
                Text(
                  'AI Assistant',
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: AppColors.navy,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const Spacer(),
                Container(
                  width: 7,
                  height: 7,
                  decoration: const BoxDecoration(
                    color: AppColors.success,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 5),
                Text(
                  'Mock',
                  style: Theme.of(context).textTheme.labelSmall
                      ?.copyWith(color: const Color(0xFF667085)),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView(
              controller: _scrollController,
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 6),
              children: [
                for (final message in widget.messages)
                  Align(
                    alignment: message.isUser
                        ? Alignment.centerRight
                        : Alignment.centerLeft,
                    child: _ChatBubble(message: message),
                  ),
                if (widget.isSending)
                  const Align(
                    alignment: Alignment.centerLeft,
                    child: _TypingBubble(),
                  ),
              ],
            ),
          ),
          DecoratedBox(
            decoration: const BoxDecoration(
              color: Colors.white,
              border: Border(top: BorderSide(color: AppColors.border)),
            ),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(10, 7, 8, 8),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      key: const Key('chat-input'),
                      controller: _messageController,
                      focusNode: widget.inputFocusNode,
                      enabled: widget.enabled && !widget.isSending,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _submit(),
                      decoration: InputDecoration(
                        hintText: widget.enabled
                            ? 'Ask a question…'
                            : 'Chat paused for disclaimer',
                        isDense: true,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 13,
                          vertical: 10,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 7),
                  IconButton.filled(
                    key: const Key('send-chat'),
                    onPressed: widget.enabled && !widget.isSending
                        ? _submit
                        : null,
                    tooltip: 'Send message',
                    style: IconButton.styleFrom(
                      backgroundColor: AppColors.blue,
                      foregroundColor: Colors.white,
                    ),
                    icon: const Icon(Icons.arrow_upward_rounded, size: 20),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ChatBubble extends StatelessWidget {
  const _ChatBubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: message.isUser ? 'You said' : 'Assistant said',
      child: Container(
        constraints: const BoxConstraints(maxWidth: 440),
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: message.isUser
              ? const Color(0xFFDDEEFF)
              : const Color(0xFFF0F2F5),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(13),
            topRight: const Radius.circular(13),
            bottomLeft: Radius.circular(message.isUser ? 13 : 3),
            bottomRight: Radius.circular(message.isUser ? 3 : 13),
          ),
        ),
        child: Text(message.text, style: Theme.of(context).textTheme.bodySmall),
      ),
    );
  }
}

class _TypingBubble extends StatelessWidget {
  const _TypingBubble();

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFF0F2F5),
        borderRadius: BorderRadius.circular(13),
      ),
      child: const SizedBox(
        width: 14,
        height: 14,
        child: CircularProgressIndicator(strokeWidth: 2),
      ),
    );
  }
}
