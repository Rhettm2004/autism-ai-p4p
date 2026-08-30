import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app.dart';
import '../models/screening_models.dart';
import '../state/screening_controller.dart';

const List<String> ethnicityOptions = [
  'Aboriginal',
  'Asian',
  'Black',
  'Hispanic',
  'Latino',
  'Maori',
  'Middle Eastern',
  'Mixed',
  'Native Indian',
  'Pacifica',
  'South Asian',
  'White European',
  'Others',
];

// TODO(ethnicity): "Test Ethnicity" appeared in the reference implementation,
// but its meaning must be confirmed before any production inclusion. It is
// deliberately not shown in this UI.

class StageCardFrame extends StatelessWidget {
  const StageCardFrame({
    super.key,
    required this.title,
    required this.child,
    this.subtitle,
    this.leading,
  });

  final String title;
  final String? subtitle;
  final Widget? leading;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (leading != null) ...[leading!, const SizedBox(width: 12)],
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      if (subtitle != null) ...[
                        const SizedBox(height: 6),
                        Text(
                          subtitle!,
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            child,
          ],
        ),
      ),
    );
  }
}

class WelcomeCard extends StatelessWidget {
  const WelcomeCard({super.key, required this.onStart});

  final VoidCallback onStart;

  @override
  Widget build(BuildContext context) {
    return StageCardFrame(
      title: 'Autism AI Assistant',
      subtitle: 'AI-supported autism screening with conversational guidance.',
      leading: Container(
        width: 52,
        height: 52,
        decoration: const BoxDecoration(
          color: AppColors.softBlue,
          shape: BoxShape.circle,
        ),
        child: const Icon(
          Icons.psychology_alt_outlined,
          color: AppColors.blue,
          size: 30,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FilledButton.icon(
            key: const Key('start-screening'),
            onPressed: onStart,
            icon: const Icon(Icons.arrow_forward_rounded),
            label: const Text('Start Screening'),
          ),
        ],
      ),
    );
  }
}

class ToddlerCheckCard extends StatefulWidget {
  const ToddlerCheckCard({
    super.key,
    required this.initialValue,
    required this.onContinue,
  });

  final bool? initialValue;
  final ValueChanged<bool> onContinue;

  @override
  State<ToddlerCheckCard> createState() => _ToddlerCheckCardState();
}

class _ToddlerCheckCardState extends State<ToddlerCheckCard> {
  bool? _selected;
  bool _showError = false;

  @override
  void initState() {
    super.initState();
    _selected = widget.initialValue;
  }

  @override
  Widget build(BuildContext context) {
    return StageCardFrame(
      title: 'Choose the age pathway',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Are you taking the test for a toddler less than 36 months old?',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 14),
          _OptionTile(
            label: 'Yes',
            selected: _selected == true,
            onTap: () => setState(() {
              _selected = true;
              _showError = false;
            }),
          ),
          const SizedBox(height: 8),
          _OptionTile(
            label: 'No',
            selected: _selected == false,
            onTap: () => setState(() {
              _selected = false;
              _showError = false;
            }),
          ),
          if (_showError) ...[
            const SizedBox(height: 12),
            const _InlineError('Select Yes or No to continue.'),
          ],
          const SizedBox(height: 18),
          FilledButton(
            onPressed: () {
              if (_selected == null) {
                setState(() => _showError = true);
                return;
              }
              widget.onContinue(_selected!);
            },
            child: const Text('Continue'),
          ),
        ],
      ),
    );
  }
}

class RespondentDetailsCard extends StatefulWidget {
  const RespondentDetailsCard({
    super.key,
    required this.controller,
    required this.onBack,
  });

  final ScreeningController controller;
  final VoidCallback onBack;

  @override
  State<RespondentDetailsCard> createState() => _RespondentDetailsCardState();
}

class _RespondentDetailsCardState extends State<RespondentDetailsCard> {
  late final TextEditingController _ageController;
  String? _gender;
  String? _ethnicity;

  @override
  void initState() {
    super.initState();
    final respondent = widget.controller.respondent;
    _ageController = TextEditingController(text: respondent.age?.toString());
    _gender = respondent.gender;
    _ethnicity = respondent.ethnicity;
  }

  @override
  void dispose() {
    _ageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isToddler = widget.controller.respondent.isToddler == true;
    return StageCardFrame(
      title: 'Respondent details',
      subtitle: isToddler
          ? 'Enter the toddler’s age in completed months.'
          : 'Enter the respondent’s age in completed years.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _FieldLabel('Gender'),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _OptionTile(
                  label: 'Male',
                  selected: _gender == 'Male',
                  onTap: () {
                    setState(() => _gender = 'Male');
                    widget.controller.setRespondentGender('Male');
                  },
                  compact: true,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _OptionTile(
                  label: 'Female',
                  selected: _gender == 'Female',
                  onTap: () {
                    setState(() => _gender = 'Female');
                    widget.controller.setRespondentGender('Female');
                  },
                  compact: true,
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          const _FieldLabel('Ethnicity'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            key: const Key('ethnicity-dropdown'),
            initialValue: _ethnicity,
            isExpanded: true,
            hint: const Text('Select ethnicity'),
            items: ethnicityOptions
                .map(
                  (value) => DropdownMenuItem(value: value, child: Text(value)),
                )
                .toList(),
            onChanged: (value) {
              setState(() => _ethnicity = value);
              widget.controller.setRespondentEthnicity(value);
            },
          ),
          const SizedBox(height: 18),
          Row(
            children: [
              _FieldLabel('Age (${isToddler ? 'months' : 'years'})'),
              const SizedBox(width: 4),
              IconButton(
                key: const Key('age-help'),
                onPressed: () => _showAgeHelp(context, isToddler),
                tooltip: 'Age help',
                visualDensity: VisualDensity.compact,
                icon: const Icon(
                  Icons.info_outline_rounded,
                  size: 20,
                  color: AppColors.blue,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          TextField(
            key: const Key('age-field'),
            controller: _ageController,
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            onChanged: (value) =>
                widget.controller.setRespondentAge(int.tryParse(value)),
            decoration: InputDecoration(
              hintText: isToddler ? 'e.g. 24' : 'e.g. 16',
              suffixText: isToddler ? 'Months' : 'Years',
            ),
          ),
          if (widget.controller.errorMessage != null) ...[
            const SizedBox(height: 12),
            _InlineError(widget.controller.errorMessage!),
          ],
          const SizedBox(height: 18),
          _NavigationButtons(
            onBack: widget.onBack,
            onNext: () => widget.controller.submitRespondentDetails(
              gender: _gender,
              ethnicity: _ethnicity,
              ageText: _ageController.text,
            ),
          ),
        ],
      ),
    );
  }

  void _showAgeHelp(BuildContext context, bool isToddler) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Age guidance'),
        content: Text(
          isToddler
              ? 'Enter the toddler’s age in completed months. Valid ages are 18 to under 36 months.'
              : 'Enter the respondent’s age in completed years. Valid ages are 3 to 80 years.',
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
}

class BackgroundQuestionsCard extends StatelessWidget {
  const BackgroundQuestionsCard({
    super.key,
    required this.controller,
    required this.onBack,
  });

  final ScreeningController controller;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    final completedByOptions = [
      if (controller.respondent.isToddler != true) 'Myself',
      'Parent',
      'Family Member',
      'Caregiver',
      'Other',
    ];

    return StageCardFrame(
      title: 'Background questions',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _YesNoQuestion(
            question: 'Were you born with jaundice?',
            value: controller.background.jaundice,
            onChanged: controller.setJaundice,
          ),
          const SizedBox(height: 20),
          _YesNoQuestion(
            question: 'Has anyone in your immediate family been diagnosed with autism?',
            value: controller.background.familyAutismHistory,
            onChanged: controller.setFamilyHistory,
          ),
          const SizedBox(height: 20),
          const _FieldLabel('Who is completing the test?'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            initialValue:
                completedByOptions.contains(controller.background.completedBy)
                ? controller.background.completedBy
                : null,
            isExpanded: true,
            hint: const Text('Select relationship'),
            items: completedByOptions
                .map(
                  (value) => DropdownMenuItem(value: value, child: Text(value)),
                )
                .toList(),
            onChanged: controller.setCompletedBy,
          ),
          if (controller.errorMessage != null) ...[
            const SizedBox(height: 12),
            _InlineError(controller.errorMessage!),
          ],
          const SizedBox(height: 18),
          _NavigationButtons(
            onBack: onBack,
            onNext: controller.submitBackgroundDetails,
          ),
        ],
      ),
    );
  }
}

class ScreeningQuestionCard extends StatelessWidget {
  const ScreeningQuestionCard({super.key, required this.controller});

  final ScreeningController controller;

  @override
  Widget build(BuildContext context) {
    final question = controller.currentQuestion!;
    final selected = controller.behaviouralAnswers[question.id];

    return StageCardFrame(
      title:
          'Question ${controller.currentQuestionIndex + 1} of ${controller.questions.length}',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _QuestionDots(
            count: controller.questions.length,
            current: controller.currentQuestionIndex,
            answeredIds: controller.behaviouralAnswers.keys.toSet(),
            questions: controller.questions,
          ),
          const SizedBox(height: 18),
          Text(question.text, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 16),
          for (final option in question.options) ...[
            _OptionTile(
              label: option,
              selected: selected == option,
              onTap: () => controller.answerCurrentQuestion(option),
            ),
            const SizedBox(height: 8),
          ],
          if (controller.errorMessage != null) ...[
            const SizedBox(height: 4),
            _InlineError(controller.errorMessage!),
          ],
          const SizedBox(height: 12),
          _NavigationButtons(
            onBack: controller.previousQuestion,
            onNext: controller.nextQuestion,
            nextLabel: controller.editingFromReview
                ? 'Save & return'
                : controller.currentQuestionIndex ==
                      controller.questions.length - 1
                ? 'Review answers'
                : 'Next',
          ),
          const SizedBox(height: 12),
          const _NoticeBox(
            icon: Icons.info_outline_rounded,
            text: 'The assistant can clarify wording, but it will not select an answer for you.',
          ),
        ],
      ),
    );
  }
}

class _QuestionDots extends StatelessWidget {
  const _QuestionDots({
    required this.count,
    required this.current,
    required this.answeredIds,
    required this.questions,
  });

  final int count;
  final int current;
  final Set<String> answeredIds;
  final List<ScreeningQuestion> questions;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: '${current + 1} of $count questions',
      child: Row(
        children: List.generate(count, (index) {
          final isCurrent = index == current;
          final answered = answeredIds.contains(questions[index].id);
          return Expanded(
            child: Container(
              height: isCurrent ? 8 : 5,
              margin: EdgeInsets.only(right: index == count - 1 ? 0 : 4),
              decoration: BoxDecoration(
                color: isCurrent
                    ? AppColors.blue
                    : answered
                    ? const Color(0xFF82ACE9)
                    : AppColors.border,
                borderRadius: BorderRadius.circular(10),
              ),
            ),
          );
        }),
      ),
    );
  }
}

class ReviewAnswersCard extends StatelessWidget {
  const ReviewAnswersCard({
    super.key,
    required this.controller,
    required this.onBack,
  });

  final ScreeningController controller;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    return StageCardFrame(
      title: 'Review your answers',
      subtitle:
          'Select the edit button to revisit any response before submitting.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (var index = 0; index < controller.questions.length; index++) ...[
            _ReviewAnswerRow(
              questionNumber: index + 1,
              question: controller.questions[index],
              answer:
                  controller.behaviouralAnswers[controller
                      .questions[index]
                      .id] ??
                  'Not answered',
              onEdit: () => controller.goToQuestion(index),
            ),
            if (index != controller.questions.length - 1)
              const SizedBox(height: 8),
          ],
          if (controller.errorMessage != null) ...[
            const SizedBox(height: 12),
            _InlineError(controller.errorMessage!),
          ],
          const SizedBox(height: 18),
          _NavigationButtons(
            onBack: onBack,
            onNext: controller.submitScreening,
            nextLabel: 'Submit Screening',
          ),
        ],
      ),
    );
  }
}

class _ReviewAnswerRow extends StatelessWidget {
  const _ReviewAnswerRow({
    required this.questionNumber,
    required this.question,
    required this.answer,
    required this.onEdit,
  });

  final int questionNumber;
  final ScreeningQuestion question;
  final String answer;
  final VoidCallback onEdit;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 4, 10),
      decoration: BoxDecoration(
        color: const Color(0xFFFAFBFD),
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 30,
            height: 30,
            alignment: Alignment.center,
            decoration: const BoxDecoration(
              color: AppColors.softBlue,
              shape: BoxShape.circle,
            ),
            child: Text(
              '$questionNumber',
              style: const TextStyle(
                color: AppColors.blue,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  question.text,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppColors.navy,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  answer,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppColors.blue,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: onEdit,
            tooltip: 'Edit question $questionNumber',
            icon: const Icon(Icons.edit_outlined, color: AppColors.blue),
          ),
        ],
      ),
    );
  }
}

class DisclaimerOverlay extends StatelessWidget {
  const DisclaimerOverlay({super.key, required this.onContinue});

  final Future<void> Function() onContinue;

  @override
  Widget build(BuildContext context) {
    // Leave the 58px header and 4px progress bar available so its contextual
    // info action still works while the disclaimer blocks the workspace.
    return Positioned(
      top: 62,
      left: 0,
      right: 0,
      bottom: 0,
      child: ColoredBox(
        color: const Color(0x990B1633),
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Material(
              color: Colors.white,
              elevation: 12,
              shadowColor: Colors.black45,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
                side: const BorderSide(color: AppColors.blue),
              ),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 480),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.health_and_safety_outlined,
                        size: 42,
                        color: AppColors.blue,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Important disclaimer',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 14),
                      Text(
                        'This app is a screening tool for research purposes. It is not a diagnosis of autism. If you have concerns, please discuss them with a qualified health professional. Anonymised data may be used for research where applicable.',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          key: const Key('accept-disclaimer'),
                          onPressed: onContinue,
                          child: const Text('I Understand & Continue'),
                        ),
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

class ResultCard extends StatelessWidget {
  const ResultCard({
    super.key,
    required this.controller,
    required this.onViewAnswers,
  });

  final ScreeningController controller;
  final VoidCallback onViewAnswers;

  @override
  Widget build(BuildContext context) {
    final result = controller.result!;
    final traitsText = result.traitsDetected
        ? 'Autistic traits were identified by the mock screening model.'
        : 'No autistic traits were identified by the mock screening model.';

    return StageCardFrame(
      title: 'Screening result',
      subtitle: 'Prototype output for interface testing only.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFFEAF8F0),
              border: Border.all(color: const Color(0xFF9BD6B4)),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              children: [
                const Icon(
                  Icons.check_circle_outline_rounded,
                  color: AppColors.success,
                  size: 44,
                ),
                const SizedBox(height: 10),
                Text(
                  traitsText,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleMedium
                      ?.copyWith(color: const Color(0xFF155E34)),
                ),
                const SizedBox(height: 8),
                Text(
                  'Mock similarity: ${result.similarityPercentage.toStringAsFixed(0)}%',
                  style: Theme.of(context).textTheme.bodyMedium
                      ?.copyWith(color: const Color(0xFF155E34)),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          const _NoticeBox(
            icon: Icons.warning_amber_rounded,
            text: 'This is a screening result and is not a clinical diagnosis.',
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              OutlinedButton.icon(
                onPressed: () =>
                    controller.sendChatMessage('Can you explain my result?'),
                icon: const Icon(Icons.chat_bubble_outline_rounded),
                label: const Text('Explain my result'),
              ),
              OutlinedButton.icon(
                onPressed: () =>
                    controller.sendChatMessage('What should I do next?'),
                icon: const Icon(Icons.route_outlined),
                label: const Text('What should I do next?'),
              ),
              OutlinedButton.icon(
                onPressed: onViewAnswers,
                icon: const Icon(Icons.fact_check_outlined),
                label: const Text('View my answers'),
              ),
            ],
          ),
          const SizedBox(height: 18),
          FilledButton(
            onPressed: controller.startValidation,
            child: const Text('Continue'),
          ),
        ],
      ),
    );
  }
}

class ValidationCard extends StatelessWidget {
  const ValidationCard({super.key, required this.controller});

  final ScreeningController controller;

  @override
  Widget build(BuildContext context) {
    return StageCardFrame(
      title: 'Research validation',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _NoticeBox(
            icon: Icons.info_outline_rounded,
            text: 'These answers are for research validation only and do not change the screening result already shown.',
          ),
          const SizedBox(height: 16),
          Text(
            'Has the respondent been formally assessed or diagnosed for autism by licensed health professionals?',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 14),
          for (final status in assessmentStatuses) ...[
            _OptionTile(
              label: status,
              selected: controller.validation.assessmentStatus == status,
              onTap: () => controller.setAssessmentStatus(status),
            ),
            const SizedBox(height: 8),
          ],
          if (controller.requiresDiagnosticTechnique) ...[
            const SizedBox(height: 12),
            const _FieldLabel(
              'What was the formal diagnostic technique used to assess the respondent?',
            ),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              initialValue: controller.validation.diagnosticTechnique,
              isExpanded: true,
              hint: const Text('Select diagnostic technique'),
              items: diagnosticTechniques
                  .map(
                    (value) =>
                        DropdownMenuItem(value: value, child: Text(value)),
                  )
                  .toList(),
              onChanged: controller.setDiagnosticTechnique,
            ),
          ],
          if (controller.errorMessage != null) ...[
            const SizedBox(height: 12),
            _InlineError(controller.errorMessage!),
          ],
          const SizedBox(height: 18),
          FilledButton(
            onPressed: controller.submitValidation,
            child: const Text('View Report'),
          ),
        ],
      ),
    );
  }
}

class ReportCard extends StatelessWidget {
  const ReportCard({
    super.key,
    required this.controller,
    required this.onDownload,
    required this.onContinueConversation,
  });

  final ScreeningController controller;
  final VoidCallback onDownload;
  final VoidCallback onContinueConversation;

  @override
  Widget build(BuildContext context) {
    final respondent = controller.respondent;
    final background = controller.background;
    final validation = controller.validation;
    final result = controller.result!;

    return StageCardFrame(
      title: 'Screening report',
      subtitle: 'Mock local summary — not a clinical report or diagnosis.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _SectionTitle('Respondent'),
          _SummaryRow(
            label: 'Age',
            value: '${respondent.age} ${respondent.ageUnit}',
          ),
          _SummaryRow(label: 'Gender', value: respondent.gender ?? '—'),
          _SummaryRow(label: 'Ethnicity', value: respondent.ethnicity ?? '—'),
          _SummaryRow(
            label: 'Questionnaire',
            value: controller.questionnaireType?.label ?? '—',
          ),
          const SizedBox(height: 16),
          const _SectionTitle('Background'),
          _SummaryRow(
            label: 'Born with jaundice',
            value: background.jaundice == true ? 'Yes' : 'No',
          ),
          _SummaryRow(
            label: 'Immediate family autism history',
            value: background.familyAutismHistory == true ? 'Yes' : 'No',
          ),
          _SummaryRow(
            label: 'Completed by',
            value: background.completedBy ?? '—',
          ),
          const SizedBox(height: 16),
          const _SectionTitle('Questions and answers'),
          for (var index = 0; index < controller.questions.length; index++)
            _ReportAnswerRow(
              number: index + 1,
              question: controller.questions[index],
              answer:
                  controller.behaviouralAnswers[controller
                      .questions[index]
                      .id] ??
                  '—',
            ),
          const SizedBox(height: 16),
          const _SectionTitle('Mock result'),
          _SummaryRow(
            label: 'Traits detected',
            value: result.traitsDetected ? 'Yes' : 'No',
          ),
          _SummaryRow(
            label: 'Mock similarity',
            value: '${result.similarityPercentage.toStringAsFixed(0)}%',
          ),
          const SizedBox(height: 16),
          const _SectionTitle('Research validation'),
          _SummaryRow(
            label: 'Formal assessment',
            value: validation.assessmentStatus ?? '—',
          ),
          if (validation.diagnosticTechnique != null)
            _SummaryRow(
              label: 'Diagnostic technique',
              value: validation.diagnosticTechnique!,
            ),
          const SizedBox(height: 16),
          const _NoticeBox(
            icon: Icons.health_and_safety_outlined,
            text: 'This screening tool is not a diagnosis. Users with concerns should speak with a qualified health professional. Anonymised data may be used for research where applicable.',
          ),
          const SizedBox(height: 18),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              OutlinedButton.icon(
                onPressed: onDownload,
                icon: const Icon(Icons.download_outlined),
                label: const Text('Download Report'),
              ),
              OutlinedButton.icon(
                onPressed: onContinueConversation,
                icon: const Icon(Icons.chat_bubble_outline_rounded),
                label: const Text('Continue Conversation'),
              ),
              FilledButton.icon(
                key: const Key('restart-screening'),
                onPressed: controller.restart,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Start New Screening'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ReportAnswerRow extends StatelessWidget {
  const _ReportAnswerRow({
    required this.number,
    required this.question,
    required this.answer,
  });

  final int number;
  final ScreeningQuestion question;
  final String answer;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Q$number. ${question.text}',
            style: Theme.of(context).textTheme.bodyMedium
                ?.copyWith(color: AppColors.navy, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 3),
          Text(
            answer,
            style: Theme.of(context).textTheme.bodyMedium
                ?.copyWith(color: AppColors.blue, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

class _SummaryRow extends StatelessWidget {
  const _SummaryRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 150,
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium
                  ?.copyWith(color: const Color(0xFF667085)),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              value,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: AppColors.navy,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        text,
        style: Theme.of(context).textTheme.titleMedium
            ?.copyWith(color: AppColors.blue),
      ),
    );
  }
}

class _YesNoQuestion extends StatelessWidget {
  const _YesNoQuestion({
    required this.question,
    required this.value,
    required this.onChanged,
  });

  final String question;
  final bool? value;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _FieldLabel(question),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _OptionTile(
                label: 'No',
                selected: value == false,
                onTap: () => onChanged(false),
                compact: true,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _OptionTile(
                label: 'Yes',
                selected: value == true,
                onTap: () => onChanged(true),
                compact: true,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _OptionTile extends StatelessWidget {
  const _OptionTile({
    required this.label,
    required this.selected,
    required this.onTap,
    this.compact = false,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      selected: selected,
      label: label,
      child: Material(
        color: selected ? AppColors.softBlue : Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(11),
          side: BorderSide(
            color: selected ? AppColors.blue : AppColors.border,
            width: selected ? 1.5 : 1,
          ),
        ),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(11),
          child: ConstrainedBox(
            constraints: const BoxConstraints(minHeight: 48),
            child: Padding(
              padding: EdgeInsets.symmetric(
                horizontal: compact ? 10 : 13,
                vertical: 10,
              ),
              child: Row(
                mainAxisAlignment: compact
                    ? MainAxisAlignment.center
                    : MainAxisAlignment.start,
                children: [
                  Container(
                    width: 19,
                    height: 19,
                    decoration: BoxDecoration(
                      color: selected ? AppColors.blue : Colors.white,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: selected
                            ? AppColors.blue
                            : const Color(0xFFAAB4C4),
                        width: 1.5,
                      ),
                    ),
                    child: selected
                        ? const Icon(Icons.check, size: 13, color: Colors.white)
                        : null,
                  ),
                  const SizedBox(width: 10),
                  Flexible(
                    child: Text(
                      label,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: selected
                            ? AppColors.navy
                            : const Color(0xFF34405C),
                        fontWeight: selected
                            ? FontWeight.w700
                            : FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _NavigationButtons extends StatelessWidget {
  const _NavigationButtons({
    required this.onBack,
    required this.onNext,
    this.nextLabel = 'Continue',
  });

  final VoidCallback onBack;
  final VoidCallback onNext;
  final String nextLabel;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: onBack,
            icon: const Icon(Icons.arrow_back_rounded),
            label: const Text('Back'),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          flex: 2,
          child: FilledButton.icon(
            onPressed: onNext,
            iconAlignment: IconAlignment.end,
            icon: const Icon(Icons.arrow_forward_rounded),
            label: Text(nextLabel),
          ),
        ),
      ],
    );
  }
}

class _FieldLabel extends StatelessWidget {
  const _FieldLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: Theme.of(context).textTheme.bodyMedium
          ?.copyWith(color: AppColors.navy, fontWeight: FontWeight.w700),
    );
  }
}

class _InlineError extends StatelessWidget {
  const _InlineError(this.message);

  final String message;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      liveRegion: true,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: const Color(0xFFFFF0F0),
          border: Border.all(color: const Color(0xFFF4B4B4)),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(
          children: [
            const Icon(Icons.error_outline_rounded, color: Color(0xFFB42318)),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                message,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: const Color(0xFF8A1C13),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NoticeBox extends StatelessWidget {
  const _NoticeBox({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.softBlue,
        borderRadius: BorderRadius.circular(11),
        border: Border.all(color: const Color(0xFFC9DCFA)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppColors.blue, size: 20),
          const SizedBox(width: 9),
          Expanded(
            child: Text(text, style: Theme.of(context).textTheme.bodySmall),
          ),
        ],
      ),
    );
  }
}
