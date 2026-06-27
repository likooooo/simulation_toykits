/**
 * Minimal Streamlit custom component bridge (browser IIFE).
 * Vendored for offline/Docker use; API compatible with streamlit-component-lib v1.
 */
(function () {
  "use strict";

  var ComponentMessageType = {
    COMPONENT_READY: "streamlit:componentReady",
    SET_COMPONENT_VALUE: "streamlit:setComponentValue",
    SET_FRAME_HEIGHT: "streamlit:setFrameHeight",
  };

  function assign(target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i];
      if (!source) continue;
      for (var key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
          target[key] = source[key];
        }
      }
    }
    return target;
  }

  var Streamlit = {
    API_VERSION: 1,
    RENDER_EVENT: "streamlit:render",
    events: new EventTarget(),
    registeredMessageListener: false,
    lastFrameHeight: null,

    setComponentReady: function () {
      if (!Streamlit.registeredMessageListener) {
        window.addEventListener("message", Streamlit.onMessageEvent);
        Streamlit.registeredMessageListener = true;
      }
      Streamlit.sendBackMsg(ComponentMessageType.COMPONENT_READY, {
        apiVersion: Streamlit.API_VERSION,
      });
    },

    setFrameHeight: function (height) {
      if (height === undefined) {
        height = document.documentElement.scrollHeight || document.body.scrollHeight;
      }
      if (height === Streamlit.lastFrameHeight) {
        return;
      }
      Streamlit.lastFrameHeight = height;
      Streamlit.sendBackMsg(ComponentMessageType.SET_FRAME_HEIGHT, { height: height });
    },

    setComponentValue: function (value) {
      Streamlit.sendBackMsg(ComponentMessageType.SET_COMPONENT_VALUE, {
        value: value,
        dataType: "json",
      });
    },

    onMessageEvent: function (event) {
      if (!event || !event.data) return;
      var type = event.data.type;
      if (type === Streamlit.RENDER_EVENT) {
        Streamlit.onRenderMessage(event.data);
      }
    },

    onRenderMessage: function (data) {
      var args = data.args;
      if (args == null) {
        args = {};
      }
      var eventData = {
        disabled: Boolean(data.disabled),
        args: args,
        theme: data.theme,
      };
      Streamlit.events.dispatchEvent(
        new CustomEvent(Streamlit.RENDER_EVENT, { detail: eventData })
      );
    },

    sendBackMsg: function (type, data) {
      window.parent.postMessage(
        assign({ isStreamlitMessage: true, type: type }, data),
        "*"
      );
    },
  };

  window.Streamlit = Streamlit;
})();
