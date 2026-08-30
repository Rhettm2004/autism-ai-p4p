import 'package:flutter/material.dart';

import '../app.dart';
import '../models/screening_models.dart';
import '../state/screening_controller.dart';
import '../widgets/app_header.dart';
import '../widgets/persistent_chat_panel.dart';
import '../widgets/stage_cards.dart';

class ScreeningPage extends StatefulWidget {
  const ScreeningPage({super.key, this.controller});

  final ScreeningController? controller;

  @override
  State<ScreeningPage> createState() => _ScreeningPageState();
}

class _ScreeningPageState extends State<ScreeningPage> {
  late final ScreeningController _controller;
  late final bool _ownsController;
  final FocusNode _chatFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ?? ScreeningController();
  }

  @override
  void dispose() {
    _chatFocusNode.dispose();
    if (_ownsController) _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 760),
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, _) {
                return Stack(
                  children: [
                    Column(
                      children: [
                        AppHeader(
                          stageLabel: _controller.stageLabel,
                          progress: _controller.overallProgress,
                          onMenuPressed: _showMenu,
                          onInfoPressed: _showInfo,
                        ),
                        Expanded(
                          child: LayoutBuilder(
                            builder: (context, constraints) {
                              final chatHeight = constraints.maxHeight < 520
                                  ? 168.0
                                  : (constraints.maxHeight * 0.31).clamp(
                                      190.0,
                                      250.0,
                                    );
                              return Padding(
                                padding: const EdgeInsets.fromLTRB(
                                  12,
                                  12,
                                  12,
                                  8,
                                ),
                                child: Column(
                                  children: [
                                    Expanded(
                                      child: SingleChildScrollView(
                                        key: PageStorageKey<String>(
                                          _contentKey,
                                        ),
                                        padding: const EdgeInsets.only(
                                          bottom: 12,
                                        ),
                                        child: AnimatedSwitcher(
                                          duration: const Duration(
                                            milliseconds: 220,
                                          ),
                                          switchInCurve: Curves.easeOut,
                                          switchOutCurve: Curves.easeIn,
                                          child: KeyedSubtree(
                                            key: ValueKey(_contentKey),
                                            child: _buildStageCard(),
                                          ),
                                        ),
                                      ),
                                    ),
                                    SizedBox(
                                      height: chatHeight,
                                      child: PersistentChatPanel(
                                        messages: _controller.chatMessages,
                                        onSend: _controller.sendChatMessage,
                                        enabled: _controller.chatEnabled,
                                        isSending: _controller.isSendingChat,
                                        inputFocusNode: _chatFocusNode,
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
                        ),
                      ],
                    ),
                    if (_controller.stage == ScreeningStage.disclaimer)
                      DisclaimerOverlay(
                        onContinue: _controller.acknowledgeDisclaimer,
                      ),
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  String get _contentKey {
    if (_controller.stage == ScreeningStage.behaviouralQuestions) {
      return '${_controller.stage.name}-${_controller.currentQuestionIndex}';
    }
    if (_controller.stage == ScreeningStage.disclaimer) {
      return ScreeningStage.review.name;
    }
    return _controller.stage.name;
  }

  Widget _buildStageCard() {
    return switch (_controller.stage) {
      ScreeningStage.welcome => WelcomeCard(
        onStart: _controller.startScreening,
      ),
      ScreeningStage.toddlerCheck => ToddlerCheckCard(
        initialValue: _controller.respondent.isToddler,
        onContinue: _controller.chooseToddlerPath,
      ),
      ScreeningStage.respondentDetails => RespondentDetailsCard(
        controller: _controller,
        onBack: _controller.backToToddlerCheck,
      ),
      ScreeningStage.backgroundQuestions => BackgroundQuestionsCard(
        controller: _controller,
        onBack: _controller.backToRespondentDetails,
      ),
      ScreeningStage.behaviouralQuestions => ScreeningQuestionCard(
        controller: _controller,
      ),
      ScreeningStage.review || ScreeningStage.disclaimer => ReviewAnswersCard(
        controller: _controller,
        onBack: _controller.backToLastQuestion,
      ),
      ScreeningStage.result => ResultCard(
        controller: _controller,
        onViewAnswers: _showAnswers,
      ),
      ScreeningStage.validation => ValidationCard(controller: _controller),
      ScreeningStage.report => ReportCard(
        controller: _controller,
        onDownload: _showDownloadPlaceholder,
        onContinueConversation: _focusChat,
      ),
    };
  }

  void _focusChat() {
    _chatFocusNode.requestFocus();
  }

  void _showMenu() {
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Autism AI', style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 8),
              const Text(
                'Flutter shell prototype. Screening responses and chat remain in memory only.',
              ),
              if (_controller.stage != ScreeningStage.welcome) ...[
                const SizedBox(height: 18),
                OutlinedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                    _confirmRestart();
                  },
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('Start New Screening'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  void _showInfo() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About this prototype'),
        content: const Text(
          'This phase demonstrates the local screening flow and persistent assistant interface. The chatbot and screening result are mocked. No model API, CNN backend, RAG system, or production storage is connected.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showAnswers() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: FractionallySizedBox(
          heightFactor: 0.8,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Your answers',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                ),
              ),
              const Divider(height: 1),
              Expanded(
                child: ListView.separated(
                  padding: const EdgeInsets.all(20),
                  itemCount: _controller.questions.length,
                  separatorBuilder: (_, _) => const Divider(height: 24),
                  itemBuilder: (context, index) {
                    final question = _controller.questions[index];
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Q${index + 1}. ${question.text}',
                          style: Theme.of(context).textTheme.bodyMedium
                              ?.copyWith(
                                color: AppColors.navy,
                                fontWeight: FontWeight.w700,
                              ),
                        ),
                        const SizedBox(height: 5),
                        Text(
                          _controller.behaviouralAnswers[question.id] ?? '—',
                          style: Theme.of(context).textTheme.bodyMedium
                              ?.copyWith(
                                color: AppColors.blue,
                                fontWeight: FontWeight.w700,
                              ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showDownloadPlaceholder() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        icon: const Icon(Icons.picture_as_pdf_outlined, color: AppColors.blue),
        title: const Text('Report download'),
        content: const Text('PDF generation will be added in a later phase.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _confirmRestart() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Start a new screening?'),
        content: const Text(
          'This clears the current screening answers and chat history.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(context);
              _controller.restart();
            },
            child: const Text('Start New'),
          ),
        ],
      ),
    );
  }
}
