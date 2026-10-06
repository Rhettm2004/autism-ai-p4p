import 'dart:convert';
import 'dart:async';

import 'package:flutter/material.dart';

import '../app.dart';
import '../config/app_config.dart';
import '../models/screening_models.dart';
import '../services/chat_service_factory.dart';
import '../services/report_download.dart';
import '../services/report_service.dart';
import '../services/screening_session_store.dart';
import '../services/screening_prediction_service_factory.dart';
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
  bool _isDownloadingReport = false;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller =
        widget.controller ??
        ScreeningController(
          chatService: createConfiguredChatService(),
          predictionService: createConfiguredPredictionService(),
          sessionStore: SharedPreferencesScreeningSessionStore(),
        );
    unawaited(_controller.initializeSession());
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
                          onNewScreening: _confirmRestart,
                        ),
                        Expanded(
                          child: LayoutBuilder(
                            builder: (context, constraints) {
                              if (AppConfig.chatFirstScreening) {
                                return Padding(
                                  padding: const EdgeInsets.fromLTRB(
                                    12,
                                    12,
                                    12,
                                    8,
                                  ),
                                  child: PersistentChatPanel(
                                    messages: _controller.chatMessages,
                                    onSend: _controller.sendChatMessage,
                                    enabled: _controller.chatEnabled,
                                    isSending: _controller.isSendingChat,
                                    serviceLabel: _controller.chatServiceLabel,
                                    inputFocusNode: _chatFocusNode,
                                    activeContent:
                                        _controller.stage ==
                                            ScreeningStage.welcome
                                        ? null
                                        : Column(
                                            children: [
                                              CompletedScreeningStepsCard(
                                                controller: _controller,
                                              ),
                                              if (_controller.stage.index >
                                                  ScreeningStage
                                                      .toddlerCheck
                                                      .index)
                                                const SizedBox(height: 10),
                                              _buildStageCard(),
                                            ],
                                          ),
                                    activeContentKey: _contentKey,
                                  ),
                                );
                              }
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
                                        serviceLabel:
                                            _controller.chatServiceLabel,
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
                    if (!AppConfig.chatFirstScreening &&
                        _controller.stage == ScreeningStage.disclaimer)
                      DisclaimerOverlay(
                        onContinue: _controller.acknowledgeDisclaimer,
                      ),
                    if (_controller.isSessionLoading)
                      const Positioned.fill(
                        child: ColoredBox(
                          color: AppColors.background,
                          child: Center(child: CircularProgressIndicator()),
                        ),
                      )
                    else if (_controller.hasRestorableSession)
                      _RestoreSessionPrompt(
                        onContinue: _controller.continueSavedSession,
                        onRestart: _controller.restartSavedSession,
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
      ScreeningStage.review => ReviewAnswersCard(
        controller: _controller,
        onBack: _controller.backToLastQuestion,
      ),
      ScreeningStage.disclaimer => InlineDisclaimerCard(
        onContinue: _controller.acknowledgeDisclaimer,
        isLoading: _controller.isCalculatingResult,
        errorMessage: _controller.errorMessage,
      ),
      ScreeningStage.result => ResultCard(
        controller: _controller,
        onViewAnswers: _showAnswers,
        onViewReport: _showReport,
      ),
      ScreeningStage.validation => ValidationCard(controller: _controller),
      ScreeningStage.report => ResultCard(
        controller: _controller,
        onViewAnswers: _showAnswers,
        onViewReport: _showReport,
      ),
    };
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
              Text(
                'About this prototype',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 8),
              const Text(
                'This prototype demonstrates the screening flow and persistent assistant interface. Chat uses the configured assistant service, and completed responses are sent to the EAIP-DARV screening model.',
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
    final info = _stageInfo;
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(info.title),
        content: Text(info.message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  ({String title, String message}) get _stageInfo =>
      switch (_controller.stage) {
        ScreeningStage.welcome => (
          title: 'Welcome',
          message: 'Start when you are ready. The assistant will remain available throughout the screening.',
        ),
        ScreeningStage.toddlerCheck => (
          title: 'Age pathway',
          message: 'Select Yes for a toddler aged 18 to under 36 months. Otherwise, select No.',
        ),
        ScreeningStage.respondentDetails => (
          title: 'Respondent details',
          message: 'Age selects the appropriate questionnaire. Enter the respondent’s current details as accurately as you can.',
        ),
        ScreeningStage.backgroundQuestions => (
          title: 'Background questions',
          message: 'These questions provide background context and are separate from the behavioural questionnaire.',
        ),
        ScreeningStage.behaviouralQuestions => (
          title: 'Behavioural questions',
          message: 'Choose the response that best reflects typical behaviour. The assistant can clarify wording but cannot answer for you.',
        ),
        ScreeningStage.review => (
          title: 'Review answers',
          message: 'Check all ten responses before submitting. Use an edit button to change an answer.',
        ),
        ScreeningStage.disclaimer => (
          title: 'Disclaimer',
          message: 'The screening is not a diagnosis. Read and acknowledge the disclaimer before viewing the result.',
        ),
        ScreeningStage.result => (
          title: 'Screening result',
          message: 'This EAIP-DARV output is a screening result and is not a clinical diagnosis. Discuss concerns with a health professional.',
        ),
        ScreeningStage.validation => (
          title: 'Validation',
          message: 'Formal-assessment details support later research validation and do not change the screening result already shown.',
        ),
        ScreeningStage.report => (
          title: 'Report and completion',
          message: 'Review the local summary, continue the conversation, or clear this session and start a new screening.',
        ),
      };

  void _showPopup(String title, Widget child) {
    showDialog<void>(
      context: context,
      builder: (context) => Dialog(
        insetPadding: const EdgeInsets.all(16),
        child: SizedBox(
          width: 760,
          height: MediaQuery.sizeOf(context).height * 0.85,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 8, 8),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        title,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                    ),
                    IconButton(
                      tooltip: 'Close',
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.close),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: child,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showReport() {
    _showPopup(
      'Screening report',
      StatefulBuilder(
        builder: (context, update) => ReportCard(
          controller: _controller,
          isDownloading: _isDownloadingReport,
          onDownload: () async {
            final future = _downloadReport();
            update(() {});
            await future;
            if (context.mounted) update(() {});
          },
        ),
      ),
    );
  }

  void _showAnswers() {
    final result = _controller.result!;
    final features = result.submission['features'] as Map?;
    final answers = result.submission['answers'] as List? ?? const [];
    _showPopup(
      'EAIP inputs and calculation details',
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (features == null)
            const Text(
              'No captured submission is available for this older or mock result.',
            )
          else ...[
            Text('Submitted: ${result.submittedAt?.toLocal()}'),
            const SizedBox(height: 12),
            const Text(
              'A value of 1 represents the scored response for that question. Agreement is not always scored: some questions describe abilities and others describe difficulties. The question-specific key determines the value.',
            ),
            for (var i = 0; i < answers.length; i++)
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text('Q${i + 1}. ${(answers[i] as Map)['question']}'),
                subtitle: Text('Selected: ${(answers[i] as Map)['answer']}'),
                trailing: Text('Value: ${features['Q${i + 1}']}'),
              ),
            const Divider(),
            for (final entry in features.entries.where(
              (entry) => !RegExp(r'^Q\d+$').hasMatch(entry.key.toString()),
            ))
              ListTile(
                title: Text(entry.key.toString()),
                trailing: Text(entry.value.toString()),
            ),
            const Text('Exact classifier request'),
            SelectableText(
              const JsonEncoder.withIndent('  ').convert({
                'features': <String, Object?>{
                  for (
                    var questionNumber = 1;
                    questionNumber <= 10;
                    questionNumber++
                  )
                    if (features.containsKey('Q$questionNumber'))
                      'Q$questionNumber': features['Q$questionNumber'],
                  for (final entry in features.entries)
                    if (!RegExp(r'^Q\d+$').hasMatch(entry.key.toString()))
                      entry.key.toString(): entry.value,
                },
              }),
            ),
            const SizedBox(height: 12),
            const Text(
              'Module 1 receives Sex as f=0 / m=1 and yes/no fields as 1/0. Modules 2 and 3 receive the submitted categories. Each then applies its saved preprocessing.',
            ),
          ],
          ModelCalculationDetails(result: result),
        ],
      ),
    );
  }

  Future<void> _downloadReport() async {
    if (_isDownloadingReport) return;
    setState(() => _isDownloadingReport = true);

    try {
      await _controller.flushPersistence();
      final reportData = ScreeningReportData.fromSession(
        _controller.sessionSnapshot,
      );
      final bytes = await const ReportService().generatePdf(reportData);
      await downloadReportPdf(bytes: bytes, filename: reportData.filename);

      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(content: Text('${reportData.filename} downloaded.')),
        );
    } catch (error, stackTrace) {
      debugPrint('Unable to generate or download the PDF report: $error');
      debugPrintStack(stackTrace: stackTrace);
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          const SnackBar(
            content: Text(
              'The PDF report could not be downloaded. Please try again.',
            ),
          ),
        );
    } finally {
      if (mounted) setState(() => _isDownloadingReport = false);
    }
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

class _RestoreSessionPrompt extends StatelessWidget {
  const _RestoreSessionPrompt({
    required this.onContinue,
    required this.onRestart,
  });

  final VoidCallback onContinue;
  final Future<void> Function() onRestart;

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: ColoredBox(
        color: const Color(0xB30B1633),
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Material(
              color: Colors.white,
              elevation: 12,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
              ),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 440),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.restore_rounded,
                        color: AppColors.blue,
                        size: 42,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Continue where you left off?',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'A screening session was saved locally on this device.',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 20),
                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton(
                              key: const Key('restart-saved-session'),
                              onPressed: onRestart,
                              child: const Text('Restart'),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: FilledButton(
                              key: const Key('continue-saved-session'),
                              onPressed: onContinue,
                              child: const Text('Continue'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
