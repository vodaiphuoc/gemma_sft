/**
 * Returns the current datetime for the message creation.
 */
function getCurrentTimestamp() {
	return new Date();
}

/**
 * Renders a message on the chat screen based on the given arguments.
 * This is called from the `showUserMessage` and `showBotMessage`.
 */
function renderMessageToScreen(args) {
	let messagesContainer = $('.messages');

	// init element
	let message = $(`
	<li class="message ${args.message_side}">
		<div class="text_wrapper">
			<div class="text">${args.text}</div>
			<div class="timestamp">${args.displayDate}</div>
		</div>
	</li>
	`);

	// add to parent
	messagesContainer.append(message);

	// animations
	setTimeout(function () {
		message.addClass('appeared');
	}, 0);
	messagesContainer.animate({ scrollTop: messagesContainer.prop('scrollHeight') }, 300);
}

/**
 * Clear all chatbox in `.messages`
 */
function clear_current_messages() {
    $('.messages').empty();
}

/**
 * Displays the user or agent message on the chat screen
 */
function showMessageByRole(message, role ,timestamp = null) {
    let displayDate = null;
    let message_side = null;

    if (timestamp === null) {
        displayDate = (getCurrentTimestamp()).toLocaleString('en-IN', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
        });
    } else {
        displayDate = timestamp
    }

    if (role === 'user') {
        message_side = 'right';
    } else {
        message_side = 'left';
    }

	// render user input message
	renderMessageToScreen({
		text: message,
		message_side: message_side,
        displayDate: displayDate
	});
}

