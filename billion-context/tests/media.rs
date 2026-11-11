//! What the kernel owes a message that carries media.
//!
//! The core never sees the bytes and never prices them. What it must get right
//! is what follows from knowing they exist: a message whose only content is an
//! image is not an empty message, and a message whose payload cannot be
//! restored after a fold must keep saying so once the pipeline has run.

use std::sync::Arc;

use billion_context::{
    CharsAsTokens, Config, Content, MediaMeta, MediaSource, Message, RenderStrategy, Role, Session,
    TurnInput,
};

fn image(recoverable: bool) -> MediaMeta {
    MediaMeta {
        mime: Some(Arc::from("image/png")),
        byte_len: Some(2_048),
        dimensions: Some((640, 480)),
        source: MediaSource::Inline,
        recoverable,
        handle: None,
    }
}

/// A caption-less image arrives as an empty text body plus media. The sweep that
/// drops blank prose must not take it with it.
#[test]
fn a_media_only_message_is_not_an_empty_message() {
    let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);
    let history = [
        Message::text("m1", Role::User, "look"),
        Message::new("m2", Role::User, Content::Text(Arc::from(""))).with_media(image(true)),
    ];
    let turn = session.process_turn(TurnInput {
        messages: &history,
        token_count: 10,
        render: RenderStrategy::None,
        now: 1,
    });
    assert_eq!(turn.messages.len(), 2, "the image message was dropped");
    assert!(turn.messages[1].media().is_some());

    // Contrast: the same shape without media is noise and goes.
    let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);
    let blank = [Message::new(
        "m1",
        Role::User,
        Content::Text(Arc::from("   ")),
    )];
    let turn = session.process_turn(TurnInput {
        messages: &blank,
        token_count: 1,
        render: RenderStrategy::None,
        now: 1,
    });
    assert!(turn.messages.is_empty());
}

/// The flag alone is the older spelling of the same fact, and a host that only
/// sets it must get the same protection: dropping the message would remove the
/// hole the recommender is supposed to leave alone.
#[test]
fn a_flagged_blank_message_is_kept_too() {
    let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);
    let history = [Message::new("m1", Role::User, Content::Text(Arc::from("")))
        .with_unrecoverable_payload(true)];
    let turn = session.process_turn(TurnInput {
        messages: &history,
        token_count: 1,
        render: RenderStrategy::None,
        now: 1,
    });
    assert_eq!(turn.messages.len(), 1);
    assert!(turn.messages[0].has_unrecoverable_payload());
}

/// The host hands in the facts; the pipeline must not launder them into a flag
/// that disagrees with them.
#[test]
fn unrecoverable_media_keeps_saying_so_through_the_pipeline() {
    let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);
    let history = [
        Message::new("m1", Role::User, Content::Text(Arc::from(""))).with_media(image(false)),
        Message::text("m2", Role::User, "after"),
    ];
    let turn = session.process_turn(TurnInput {
        messages: &history,
        token_count: 10,
        render: RenderStrategy::None,
        now: 1,
    });
    assert_eq!(turn.messages.len(), 2);
    assert!(turn.messages[0].has_unrecoverable_payload());
    assert!(!turn.messages[1].has_unrecoverable_payload());
}

/// Identities are what reconcile matches on, and media is part of one: a
/// resent original whose picture changed is a different message.
#[test]
fn media_is_part_of_what_a_message_is() {
    let plain = Message::text("m1", Role::User, "here");
    let with_image = plain.clone().with_media(image(true));
    let other_image = plain.clone().with_media(MediaMeta {
        dimensions: Some((1280, 720)),
        ..image(true)
    });
    assert_ne!(plain, with_image);
    assert_ne!(with_image, other_image);
}

/// A persisted state must bring the facts back, or a restart would silently
/// restore a fold that destroys an image.
#[cfg(feature = "serde")]
#[test]
fn media_survives_a_serde_round_trip() {
    let msg = Message::text("m1", Role::User, "look").with_media(MediaMeta {
        recoverable: false,
        handle: Some(Arc::from("sha256:abc")),
        ..image(false)
    });
    let json = serde_json::to_string(&msg).expect("serialise");
    let back: Message = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(back, msg);
    assert!(back.has_unrecoverable_payload());
    assert_eq!(
        back.media().and_then(|m| m.handle.as_deref()),
        Some("sha256:abc")
    );
}

/// A message written before media existed deserialises with `None`, so an old
/// snapshot keeps loading. Simulated by dropping the field from real JSON
/// rather than hand-writing a shape that could drift from the derives.
#[cfg(feature = "serde")]
#[test]
fn a_message_without_media_still_loads() {
    let msg = Message::text("m1", Role::User, "hi");
    let mut snapshot = serde_json::to_value(&msg).expect("serialise");
    let fields = snapshot.as_object_mut().expect("message is an object");
    assert!(
        fields.remove("media").is_some(),
        "the field exists, so removing it is a real pre-media snapshot"
    );
    let back: Message = serde_json::from_value(snapshot).expect("old snapshot");
    assert!(back.media().is_none());
    assert!(!back.has_unrecoverable_payload());
}
